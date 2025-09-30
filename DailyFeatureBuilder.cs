using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CsvHelper;

namespace DataProcessing;

public static class DailyFeatureBuilder
{
    public static void BuildFeatures(
        string inputPath,
        string outputPath,
        double promoDropThreshold = 0.05)
    {
        if (!File.Exists(inputPath)) throw new FileNotFoundException(inputPath);

        // --- Read input CSV into memory (case-insensitive headers) ---
        var rows = new List<AggregatedRow>();
        using (var reader = new StreamReader(inputPath))
        using (var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            if (!csvReader.Read()) throw new Exception("Input CSV is empty.");
            csvReader.ReadHeader();

            var colProduct = "product_sku";
            var colDate = "date";
            var colSold = "sold_count";
            var colAvgPrice = "avg_price";
            var colOrders = "orders_count";
            var colObserved = "is_observed";

            while (csvReader.Read())
            {
                var sku = csvReader.GetField(colProduct)?.Trim() ?? throw new InvalidOperationException(colProduct);

                var rawDate = csvReader.GetField(colDate);
                if (string.IsNullOrEmpty(rawDate) || !TryParseDate(rawDate, out var dt)) throw new InvalidOperationException(colDate);

                var sold = ParseDoubleSafe(csvReader.GetField(colSold) ?? throw new InvalidOperationException(colSold));
                var price = ParseDoubleSafe(csvReader.GetField(colAvgPrice) ?? throw new InvalidOperationException(colAvgPrice));
                var orders = ParseIntSafe(csvReader.GetField(colOrders) ?? throw new InvalidOperationException(colOrders));
                var isObserved = csvReader.GetField(colObserved) ?? throw new InvalidOperationException(colObserved);

                rows.Add(new AggregatedRow
                {
                    ProductSku = sku,
                    Date = dt.Date,
                    SoldCount = sold,
                    AvgPrice = price,
                    OrdersCount = orders,
                    IsObserved = isObserved == "1" ? 1 : 0
                });
            }
        }

        if (rows.Count == 0) throw new Exception("No valid rows read from input.");

        var perProduct = rows.GroupBy(r => r.ProductSku)
            .ToDictionary(g => g.Key, g => g.OrderBy(x => x.Date).ToList());
        
        // Compute features per product (assumes continuous dates now or will compute with gaps)
        var outputRows = new List<FeatureRow>();
        foreach (var kv in perProduct.OrderBy(k => k.Key))
        {
            var list = kv.Value;
            var n = list.Count;
            var sold = list.Select(x => x.SoldCount).ToArray();
            var priceArr = list.Select(x => x.AvgPrice).ToArray();

            // build feature rows with historical-only computations
            for (var i = 0; i < n; i++)
            {
                var fr = new FeatureRow
                {
                    ProductSku = list[i].ProductSku,
                    Date = list[i].Date,
                    SoldCount = list[i].SoldCount,
                    AvgPrice = list[i].AvgPrice.HasValue ? list[i].AvgPrice : (double?)null,
                    OrdersCount = list[i].OrdersCount,
                    IsObserved = list[i].IsObserved
                };

                fr.Lag1 = (i - 1 >= 0) ? sold[i - 1] : 0.0;
                fr.Lag7 = (i - 7 >= 0) ? sold[i - 7] : 0.0;
                fr.Lag14 = (i - 14 >= 0) ? sold[i - 14] : 0.0;

                // trailing rolling mean excluding current
                fr.Roll7Mean = (i - 1 >= 0) ? sold[Math.Max(0, i - 7)..i].Average() : 0.0;
                fr.Roll14Mean = (i - 1 >= 0) ? sold[Math.Max(0, i - 14)..i].Average() : 0.0;

                // temporal
                var dow = ((int)fr.Date.DayOfWeek + 6) % 7; // Monday=0
                var angle = 2.0 * Math.PI * dow / 7.0;
                fr.DowSin = Math.Sin(angle);
                fr.DowCos = Math.Cos(angle);
                fr.IsWeekend = dow >= 5 ? 1 : 0;
                fr.Month = fr.Date.Month;

                // price change pct: compare to last observed price (walk backward)
                double? lastObservedPrice = null;
                for (var j = i - 1; j >= 0; j--)
                {
                    if (!priceArr[j].HasValue) continue;

                    lastObservedPrice = priceArr[j]!.Value;
                    break;
                }
                if (lastObservedPrice.HasValue && fr.AvgPrice.HasValue && lastObservedPrice.Value > 0.0)
                    fr.PriceChangePct = (fr.AvgPrice.Value - lastObservedPrice.Value) / lastObservedPrice.Value;
                else
                    fr.PriceChangePct = 0.0;

                fr.PromoFlag = (lastObservedPrice.HasValue && fr.AvgPrice.HasValue && lastObservedPrice.Value > 0.0 &&
                                (fr.AvgPrice.Value - lastObservedPrice.Value) / lastObservedPrice.Value <= -promoDropThreshold) ? 1 : 0;

                outputRows.Add(fr);
            }
        }

        using (var writer = new StreamWriter(outputPath))
        using (var csvOut = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            // header
            csvOut.WriteField("product_sku");
            csvOut.WriteField("date");
            csvOut.WriteField("sold_count");
            csvOut.WriteField("avg_price");
            csvOut.WriteField("orders_count");
            csvOut.WriteField("is_observed");

            csvOut.WriteField("lag_1");
            csvOut.WriteField("lag_7");
            csvOut.WriteField("lag_14");
            csvOut.WriteField("roll7_mean");
            csvOut.WriteField("roll14_mean");

            csvOut.WriteField("dow_sin");
            csvOut.WriteField("dow_cos");
            csvOut.WriteField("is_weekend");
            csvOut.WriteField("month");

            csvOut.WriteField("price_change_pct");
            csvOut.WriteField("promo_flag");

            csvOut.NextRecord();

            foreach (var r in outputRows)
            {
                csvOut.WriteField(r.ProductSku);
                csvOut.WriteField(r.Date.ToString("yyyy-MM-dd"));
                csvOut.WriteField(r.SoldCount);
                csvOut.WriteField(r.AvgPrice.HasValue ? r.AvgPrice.Value.ToString(CultureInfo.InvariantCulture) : "");
                csvOut.WriteField(r.OrdersCount);
                csvOut.WriteField(r.IsObserved);

                csvOut.WriteField(r.Lag1);
                csvOut.WriteField(r.Lag7);
                csvOut.WriteField(r.Lag14);
                csvOut.WriteField(r.Roll7Mean);
                csvOut.WriteField(r.Roll14Mean);

                csvOut.WriteField(r.DowSin);
                csvOut.WriteField(r.DowCos);
                csvOut.WriteField(r.IsWeekend);
                csvOut.WriteField(r.Month);

                csvOut.WriteField(r.PriceChangePct);
                csvOut.WriteField(r.PromoFlag);

                csvOut.NextRecord();
            }
        }

        Console.WriteLine($"Features written to {outputPath} (rows: {outputRows.Count}, products: {perProduct.Count}");
    }

    private static double ParseDoubleSafe(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) throw new Exception($"No valid value to parsed: {raw}");
        var s = raw.Trim();
        s = Regex.Replace(s, @"[^\d\-\.,]", "");
        if (s.Contains('.') && s.Contains(',')) s = s.Replace(",", "");
        else if (s.Contains(',') && !s.Contains('.')) s = s.Replace(",", ".");
        if (double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v)) return v;
        if (double.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out v)) return v;
        throw new Exception($"Can't parsed value: {raw}");
    }

    private static int ParseIntSafe(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) throw new Exception($"No valid value to parsed: {raw}");
        if (int.TryParse(raw, NumberStyles.Any, CultureInfo.InvariantCulture, out var v)) return v;
        var s = Regex.Replace(raw, @"[^\d\-]", "");
        if (int.TryParse(s, out v)) return v;
        throw new Exception($"Can't parsed value: {raw}");
    }

    private static bool TryParseDate(string raw, out DateTime dt)
    {
        dt = default;
        if (string.IsNullOrWhiteSpace(raw)) return false;
        if (DateTime.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out dt)) return true;
        var formats = new[] { "yyyy-MM-dd", "yyyy/MM/dd", "dd/MM/yyyy", "d/M/yyyy", "dd-MM-yyyy", "d-M-yyyy", "M/d/yyyy", "MM/dd/yyyy" };
        if (DateTime.TryParseExact(raw, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out dt)) return true;
        if (DateTime.TryParse(raw, CultureInfo.CurrentCulture, DateTimeStyles.AssumeLocal, out dt)) return true;
        return false;
    }

    // Input and output row types
    private class AggregatedRow
    {
        public string ProductSku = string.Empty;
        public DateTime Date = new();
        public double SoldCount = 0;
        public double? AvgPrice = null;
        public int OrdersCount = 0;
        public int IsObserved = 0;
    }

    private class FeatureRow
    {
        public string ProductSku = string.Empty;
        public DateTime Date = new();
        public double SoldCount = 0;
        public double? AvgPrice = null;
        public int OrdersCount = 0;
        public int IsObserved = 0;

        public double Lag1 = 0;
        public double Lag7 = 0;
        public double Lag14 = 0;
        public double Roll7Mean = 0;
        public double Roll14Mean = 0;

        public double DowSin = 0;
        public double DowCos = 0;
        public int IsWeekend = 0;
        public int Month = 0;

        public double PriceChangePct = 0;
        public int PromoFlag = 0;
    }
}
