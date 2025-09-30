using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CsvHelper;

namespace DataProcessing;

public static class WeeklyFeatureBuilder
{
    public static void BuildFeatures(
        string inputPath,
        string outputPath,
        double promoDropThreshold = 0.05,
        DayOfWeek weekStartDay = DayOfWeek.Monday)
    {
        if (!File.Exists(inputPath)) throw new FileNotFoundException(inputPath);

        // Read input file and detect format
        var dailyRows = new List<DailyRow>();
        List<WeeklyAggRow> weeklyRows;

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

            // read daily aggregated rows
            while (csvReader.Read())
            {
                var sku = csvReader.GetField(colProduct)?.Trim() ?? throw new InvalidOperationException(colProduct);

                var rawDate = csvReader.GetField(colDate);
                if (string.IsNullOrWhiteSpace(rawDate) || !TryParseDate(rawDate, out var dt)) throw new InvalidOperationException(colDate);

                var sold = ParseDoubleSafe(csvReader.GetField(colSold) ?? throw new InvalidOperationException(colSold));
                var price = ParseDoubleSafe(csvReader.GetField(colAvgPrice) ?? throw new InvalidOperationException(colAvgPrice));
                var orders = ParseIntSafe(csvReader.GetField(colOrders) ?? throw new InvalidOperationException(colOrders));
                var isObserved = ParseIntSafe(csvReader.GetField(colObserved) ?? throw new InvalidOperationException(colObserved));

                dailyRows.Add(new DailyRow
                {
                    ProductSku = sku,
                    Date = dt.Date,
                    SoldCount = sold,
                    AvgPrice = price,
                    OrdersCount = orders,
                    IsObserved = isObserved
                });
            }

            // aggregate daily -> weekly (weekStartDay)
            weeklyRows = AggregateDailyToWeeklyInternal(dailyRows, weekStartDay);
        }

        if (weeklyRows.Count == 0) throw new Exception("No rows after reading/aggregation.");

        // Build per-product timeline (ordered by week start)
        var perProduct = weeklyRows.GroupBy(r => r.ProductSku)
            .ToDictionary(g => g.Key, g => g.OrderBy(x => x.WeekStart).ToList());

        // Compute features per product
        var outputRows = new List<WeeklyFeatureRow>();
        foreach (var kv in perProduct.OrderBy(k => k.Key))
        {
            var list = kv.Value;
            var n = list.Count;
            var soldArr = list.Select(x => x.SoldSum).ToArray();
            var priceArr = list.Select(x => x.AvgPrice).ToArray();

            for (var i = 0; i < n; i++)
            {
                var w = list[i];
                var fr = new WeeklyFeatureRow
                {
                    ProductSku = w.ProductSku,
                    WeekStart = w.WeekStart,
                    SoldSum = w.SoldSum,
                    AvgPrice = w.AvgPrice,
                    OrdersSum = w.OrdersSum,
                    IsObserved = w.IsObserved,
                    DaysWithSales = w.DaysWithSales,
                    // weekly lags
                    Lag1W = (i - 1 >= 0) ? soldArr[i - 1] : 0.0,
                    Lag4W = (i - 4 >= 0) ? soldArr[i - 4] : 0.0,
                    Lag12W = (i - 12 >= 0) ? soldArr[i - 12] : 0.0,
                    // trailing rolling means excluding current week
                    Roll4Mean = (i - 1 >= 0) ? soldArr[Math.Max(0, i - 4)..i].Average() : 0.0,
                    Roll12Mean = (i - 1 >= 0) ? soldArr[Math.Max(0, i - 12)..i].Average() : 0.0
                };

                // temporal features: week-of-year (sin/cos), month, quarter
                var weekOfYear = CultureInfo.InvariantCulture.Calendar.GetWeekOfYear(
                    fr.WeekStart, CalendarWeekRule.FirstFourDayWeek, DayOfWeek.Monday);
                var angle = 2.0 * Math.PI * weekOfYear / 52.0;
                fr.WoySin = Math.Sin(angle);
                fr.WoyCos = Math.Cos(angle);
                fr.Month = fr.WeekStart.Month;
                fr.Quarter = (fr.WeekStart.Month - 1) / 3 + 1;

                // price change pct: compare to last observed weekly price
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

                fr.PriceMissing = (!fr.AvgPrice.HasValue || fr.AvgPrice.GetValueOrDefault() == 0.0) ? 1 : 0;

                outputRows.Add(fr);
            }
        }

        // Write weekly features CSV
        using (var writer = new StreamWriter(outputPath))
        using (var csvOut = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            // header (ordered)
            csvOut.WriteField("product_sku");
            csvOut.WriteField("week_start");
            csvOut.WriteField("sold_sum");
            csvOut.WriteField("avg_price");
            csvOut.WriteField("orders_sum");
            csvOut.WriteField("is_observed");
            csvOut.WriteField("days_with_sales");

            csvOut.WriteField("lag_1w");
            csvOut.WriteField("lag_4w");
            csvOut.WriteField("lag_12w");
            csvOut.WriteField("roll4_mean");
            csvOut.WriteField("roll12_mean");

            csvOut.WriteField("woy_sin");
            csvOut.WriteField("woy_cos");
            csvOut.WriteField("month");
            csvOut.WriteField("quarter");

            csvOut.WriteField("price_change_pct");
            csvOut.WriteField("promo_flag");
            csvOut.WriteField("price_missing");

            csvOut.NextRecord();

            foreach (var r in outputRows)
            {
                csvOut.WriteField(r.ProductSku);
                csvOut.WriteField(r.WeekStart.ToString("yyyy-MM-dd"));
                csvOut.WriteField(r.SoldSum);
                csvOut.WriteField(r.AvgPrice.HasValue ? r.AvgPrice.Value.ToString(CultureInfo.InvariantCulture) : "");
                csvOut.WriteField(r.OrdersSum);
                csvOut.WriteField(r.IsObserved);
                csvOut.WriteField(r.DaysWithSales);

                csvOut.WriteField(r.Lag1W);
                csvOut.WriteField(r.Lag4W);
                csvOut.WriteField(r.Lag12W);
                csvOut.WriteField(r.Roll4Mean);
                csvOut.WriteField(r.Roll12Mean);

                csvOut.WriteField(r.WoySin);
                csvOut.WriteField(r.WoyCos);
                csvOut.WriteField(r.Month);
                csvOut.WriteField(r.Quarter);

                csvOut.WriteField(r.PriceChangePct);
                csvOut.WriteField(r.PromoFlag);
                csvOut.WriteField(r.PriceMissing);

                csvOut.NextRecord();
            }
        }

        Console.WriteLine($"Weekly features written to {outputPath} (rows: {outputRows.Count}, products: {perProduct.Count})");
    }

    private static List<WeeklyAggRow> AggregateDailyToWeeklyInternal(List<DailyRow> dailyRows, DayOfWeek weekStartDay)
    {
        // compute week start for a date
        DateTime WeekStart(DateTime d)
        {
            var diff = (7 + (d.DayOfWeek - weekStartDay)) % 7;
            return d.AddDays(-diff).Date;
        }

        // group daily by sku and weekStart
        var grouped = dailyRows
            .GroupBy(r => r.ProductSku)
            .ToDictionary(g => g.Key, g => g.GroupBy(r => WeekStart(r.Date))
                                             .ToDictionary(wg => wg.Key, wg => wg.ToList()));

        // compute global week range from all daily rows (aligned to week starts)
        var minDate = dailyRows.Min(r => r.Date);
        var maxDate = dailyRows.Max(r => r.Date);
        var startWeek = WeekStart(minDate);
        var endWeek = WeekStart(maxDate);

        var weeks = new List<DateTime>();
        for (var d = startWeek; d <= endWeek; d = d.AddDays(7)) weeks.Add(d);

        var outRows = new List<WeeklyAggRow>();
        foreach (var sku in grouped.Keys.OrderBy(k => k))
        {
            var weekMap = grouped[sku];
            foreach (var wk in weeks)
            {
                if (weekMap.TryGetValue(wk, out var dayList))
                {
                    var soldSum = dayList.Sum(x => x.SoldCount);
                    var ordersSum = dayList.Sum(x => x.OrdersCount);
                    var daysWithSales = dayList.Count(x => x.SoldCount > 0.0);
                    var isObs = daysWithSales > 0 ? 1 : 0;

                    // weighted avg price by sold if have sold
                    double? avgPrice = null;
                    if (soldSum > 0)
                    {
                        var numer = dayList.Where(x => x.AvgPrice.HasValue).Sum(x => (x.AvgPrice ?? 0.0) * x.SoldCount);
                        if (numer > 0) avgPrice = numer / soldSum;
                        else
                        {
                            var prices = dayList.Where(x => x.AvgPrice.HasValue).Select(x => x.AvgPrice!.Value).ToArray();
                            if (prices.Length > 0) avgPrice = prices.Average();
                        }
                    }
                    else
                    {
                        var prices = dayList.Where(x => x.AvgPrice.HasValue).Select(x => x.AvgPrice!.Value).ToArray();
                        if (prices.Length > 0) avgPrice = prices.Average();
                    }

                    outRows.Add(new WeeklyAggRow
                    {
                        ProductSku = sku,
                        WeekStart = wk,
                        SoldSum = soldSum,
                        AvgPrice = avgPrice,
                        OrdersSum = ordersSum,
                        IsObserved = isObs,
                        DaysWithSales = daysWithSales
                    });
                }
                else
                {
                    outRows.Add(new WeeklyAggRow
                    {
                        ProductSku = sku,
                        WeekStart = wk,
                        SoldSum = 0.0,
                        AvgPrice = null,
                        OrdersSum = 0,
                        IsObserved = 0,
                        DaysWithSales = 0
                    });
                }
            }
        }

        // forward/backfill weekly avg price per sku for continuity (same approach as daily)
        var outBySku = outRows.GroupBy(r => r.ProductSku).ToDictionary(g => g.Key, g => g.OrderBy(x => x.WeekStart).ToList());
        var final = new List<WeeklyAggRow>();
        foreach (var sku in outBySku.Keys.OrderBy(k => k))
        {
            var list = outBySku[sku];
            double? last = null;
            foreach (var t in list)
            {
                if (t.AvgPrice.HasValue) last = t.AvgPrice;
                else if (last.HasValue) t.AvgPrice = last;
            }
            last = null;
            for (var i = list.Count - 1; i >= 0; i--)
            {
                if (list[i].AvgPrice.HasValue) last = list[i].AvgPrice;
                else if (last.HasValue) list[i].AvgPrice = last;
            }

            final.AddRange(list);
        }

        return final;
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
        raw = raw.Trim();
        if (DateTime.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out dt)) return true;
        var formats = new[] { "yyyy-MM-dd", "yyyy/MM/dd", "dd/MM/yyyy", "d/M/yyyy", "dd-MM-yyyy", "d-M-yyyy", "M/d/yyyy", "MM/dd/yyyy" };
        if (DateTime.TryParseExact(raw, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out dt)) return true;
        if (DateTime.TryParse(raw, CultureInfo.CurrentCulture, DateTimeStyles.AssumeLocal, out dt)) return true;
        return false;
    }

    private class DailyRow
    {
        public string ProductSku = string.Empty;
        public DateTime Date = new();
        public double SoldCount = 0;
        public double? AvgPrice = null;
        public int OrdersCount = 0;
        public int IsObserved = 0;
    }

    private class WeeklyAggRow
    {
        public string ProductSku = string.Empty;
        public DateTime WeekStart = new();
        public double SoldSum = 0;
        public double? AvgPrice = null;
        public int OrdersSum = 0;
        public int IsObserved = 0;
        public int DaysWithSales = 0;
    }

    private class WeeklyFeatureRow
    {
        public string ProductSku = string.Empty;
        public DateTime WeekStart = new();
        public double SoldSum = 0;
        public double? AvgPrice = null;
        public int OrdersSum = 0;
        public int IsObserved = 0;
        public int DaysWithSales = 0;

        public double Lag1W = 0;
        public double Lag4W = 0;
        public double Lag12W = 0;
        public double Roll4Mean = 0;
        public double Roll12Mean = 0;

        public double WoySin = 0;
        public double WoyCos = 0;
        public int Month = 0;
        public int Quarter = 0;

        public double PriceChangePct = 0;
        public int PromoFlag = 0;
        public int PriceMissing = 0;
    }
}
