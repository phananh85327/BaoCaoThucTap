using CsvHelper;
using System.Globalization;
using System.Text.RegularExpressions;

namespace DataProcessing;

public class OrderAggregator
{
    public static void AggregateToProductDaily(
        string inputPath,
        string outputPath,
        string orderId = "Mã đơn hàng",
        string productSkuColumn = "SKU sản phẩm",
        string productNameColumn = "Tên sản phẩm",
        string dateColumn = "Ngày đặt hàng",
        string qtyColumn = "Số lượng",
        string returnColumn = "Số lượng sản phẩm được hoàn trả",
        string priceColumn = "Giá ưu đãi")
    {
        if (!File.Exists(inputPath)) throw new FileNotFoundException("Input file not found", inputPath);

        // Read file and header
        using var reader = new StreamReader(inputPath);
        using var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture);

        if (!csvReader.Read()) throw new Exception("Empty input CSV.");
        csvReader.ReadHeader();

        // Read rows into memory (safe for small/medium files)
        var rows = new List<OrderRow>();
        do
        {
            var row = new OrderRow
            {
                OrderId = csvReader.GetField(orderId)?.Trim() ?? throw new InvalidOperationException(orderId)
            };

            if (row.OrderId == orderId) continue;

            // product sku fallback to name
            row.ProductSku = csvReader.GetField(productSkuColumn)?.Trim() ?? throw new InvalidOperationException(productSkuColumn);

            // parse date
            var rawDate = csvReader.GetField(dateColumn);
            if (rawDate == null || !TryParseDate(rawDate, out var dt)) throw new InvalidOperationException(dateColumn);
            row.OrderDate = dt.Date;

            // quantity
            row.Qty = ParseDecimalRobust(csvReader.GetField(qtyColumn) ?? throw new InvalidOperationException(qtyColumn));

            // returns
            row.Returns = ParseDecimalRobust(csvReader.GetField(returnColumn) ?? throw new InvalidOperationException(returnColumn));

            // net qty
            row.NetQty = row.Qty - row.Returns;

            // price
            row.Price = ParseDecimalRobust(csvReader.GetField(priceColumn) ?? throw new InvalidOperationException(priceColumn));

            rows.Add(row);
        } while (csvReader.Read());

        if (rows.Count == 0) throw new Exception("No valid rows parsed from CSV.");

        // Aggregate per product × date
        var aggregated = rows
            .GroupBy(r => new { r.ProductSku, r.OrderDate })
            .Select(g =>
            {
                var prices = g.Select(x => x.Price).ToArray();
                return new AggRow
                {
                    ProductSku = g.Key.ProductSku,
                    Date = g.Key.OrderDate,
                    SoldCount = g.Sum(x => (double?)x.NetQty ?? 0.0),
                    AvgPrice = prices.Length > 0 ? (double?)prices.Average() : null,
                    OrdersCount = g.Select(x => x.OrderId).Where(x => !string.IsNullOrEmpty(x)).Distinct().Count()
                };
            })
            .ToList();

        // Determine global date range
        var minDate = aggregated.Min(a => a.Date);
        var maxDate = aggregated.Max(a => a.Date);

        // Build full daily range
        var fullDates = Enumerable.Range(0, (int)(maxDate - minDate).TotalDays + 1)
                                  .Select(i => minDate.AddDays(i))
                                  .ToArray();

        // For each product, reindex to fullDates and fill missing days
        var perProduct = aggregated.GroupBy(a => a.ProductSku)
                                   .ToDictionary(g => g.Key, g => g.ToDictionary(x => x.Date, x => x));

        var outputRows = new List<OutRow>();
        foreach (var product in perProduct.Keys.OrderBy(k => k))
        {
            // create array of days
            var dayList = new List<OutRow>();
            foreach (var d in fullDates)
            {
                if (perProduct[product].TryGetValue(d, out var aggRow))
                {
                    dayList.Add(new OutRow
                    {
                        ProductSku = product,
                        Date = d,
                        SoldCount = aggRow.SoldCount,
                        AvgPrice = aggRow.AvgPrice,
                        OrdersCount = aggRow.OrdersCount,
                        IsObserved = 1
                    });
                }
                else
                {
                    dayList.Add(new OutRow
                    {
                        ProductSku = product,
                        Date = d,
                        SoldCount = 0.0,
                        AvgPrice = null,
                        OrdersCount = 0,
                        IsObserved = 0
                    });
                }
            }

            // forward fill avg_price then backfill
            double? last = null;
            foreach (var t in dayList)
            {
                if (t.AvgPrice.HasValue) last = t.AvgPrice;
                else if (last.HasValue) t.AvgPrice = last;
            }
            // backfill remaining (from end)
            last = null;
            for (var i = dayList.Count - 1; i >= 0; i--)
            {
                if (dayList[i].AvgPrice.HasValue) last = dayList[i].AvgPrice;
                else if (last.HasValue) dayList[i].AvgPrice = last;
            }

            outputRows.AddRange(dayList);
        }

        // Write CSV
        using var writer = new StreamWriter(outputPath);
        using var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // header
        csvWriter.WriteField("product_sku");
        csvWriter.WriteField("date");
        csvWriter.WriteField("sold_count");
        csvWriter.WriteField("avg_price");
        csvWriter.WriteField("orders_count");
        csvWriter.WriteField("is_observed");
        csvWriter.NextRecord();

        foreach (var r in outputRows)
        {
            csvWriter.WriteField(r.ProductSku);
            csvWriter.WriteField(r.Date.ToString("yyyy-MM-dd"));
            csvWriter.WriteField(r.SoldCount);
            // write avg_price empty if null
            if (r.AvgPrice.HasValue)
                csvWriter.WriteField(r.AvgPrice.Value);
            else
                csvWriter.WriteField(string.Empty);
            csvWriter.WriteField(r.OrdersCount);
            csvWriter.WriteField(r.IsObserved);
            csvWriter.NextRecord();
        }
    }

    // Helper: parse messy numeric strings robustly
    private static decimal ParseDecimalRobust(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) throw new Exception($"No valid value to parsed: {raw}");

        raw = raw.Trim();

        // remove non-digit except dot, comma, minus
        var s = Regex.Replace(raw, @"[^\d\-,\.]", "");

        // If both dot and comma exist, assume commas are thousands separators -> remove commas
        if (s.Contains('.') && s.Contains(','))
            s = s.Replace(",", "");

        // If only comma exists (no dot) -> treat comma as decimal separator
        else if (s.Contains(',') && !s.Contains('.'))
            s = s.Replace(",", ".");

        // Remove leftover spaces
        s = s.Replace(" ", "");

        if (decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var val))
            return val;

        // last resort try current culture
        if (decimal.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out val))
            return val;

        throw new Exception($"Can't parsed value: {raw}");
    }

    // Helper: parse dates with leniency
    private static bool TryParseDate(string raw, out DateTime dt)
    {
        dt = default;
        if (string.IsNullOrWhiteSpace(raw)) return false;
        raw = raw.Trim();

        // Try invariant culture first
        if (DateTime.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out dt))
            return true;

        // Try common formats (day-month-year and year-month-day)
        var formats = new[]
        {
            "yyyy-MM-dd", "yyyy/MM/dd", "dd/MM/yyyy", "d/M/yyyy", "dd-MM-yyyy", "d-M-yyyy", "M/d/yyyy", "MM/dd/yyyy"
        };
        if (DateTime.TryParseExact(raw, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out dt))
            return true;

        // Try current culture
        if (DateTime.TryParse(raw, CultureInfo.CurrentCulture, DateTimeStyles.AssumeLocal, out dt))
            return true;

        return false;
    }

    // Internal row types
    private class OrderRow
    {
        public string OrderId = string.Empty;
        public string ProductSku = string.Empty;
        public DateTime OrderDate = new();
        public decimal Qty = 0;
        public decimal Returns = 0;
        public decimal NetQty = 0;
        public decimal Price = 0;
    }

    private class AggRow
    {
        public string ProductSku = string.Empty;
        public DateTime Date = new();
        public double SoldCount = 0;
        public double? AvgPrice = null;
        public int OrdersCount = 0;
    }

    private class OutRow
    {
        public string ProductSku = string.Empty;
        public DateTime Date = new();
        public double SoldCount = 0;
        public double? AvgPrice = null;
        public int OrdersCount = 0;
        public int IsObserved;
    }
}
