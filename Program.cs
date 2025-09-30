using CsvHelper;
using System.Globalization;

namespace DataProcessing;

class Program
{
    private const string InputFolder = @"D:\00 - Project\LSTM\DataProcessing\Input";
    private const string OutputFile = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.csv";
    private const string FilterOutput = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.filter.csv";
    private const string LabeledOutput = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.label.csv";
    private const string AggregatedOutput = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.aggregated.csv";
    private const string DailyBuildFeaturesOutput = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.daily_build_features.csv";
    private const string WeeklyBuildFeaturesOutput = @"D:\00 - Project\LSTM\DataProcessing\Output\Order.all.weekly_build_features.csv";

    public static void FilterCsvByColumns(string inputPath, string outputPath, string[] selectedColumns, bool ignoreMissing = false)
    {
        if (selectedColumns == null || selectedColumns.Length == 0)
            throw new ArgumentException("selectedColumns must contain at least one column name.");

        if (!File.Exists(inputPath))
            throw new ArgumentException("inputPath isn't valid.");

        using var reader = new StreamReader(inputPath);
        using var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture);

        // Read header
        csvReader.Read();
        csvReader.ReadHeader();
        var header = csvReader.HeaderRecord;

        if (header == null) throw new Exception("Empty header");

        var finalHeaders = selectedColumns.Select(col => header.Contains(col) ? col : null).ToArray();
        if (!ignoreMissing)
        {
            var missing = selectedColumns.Where((c, i) => finalHeaders[i] == null).ToArray();
            if (missing.Length != 0) throw new Exception($"Missing columns in input CSV: {string.Join(", ", missing)}");
        }

        using var writer = new StreamWriter(outputPath);
        using var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // Write header (only the selected columns that exist)
        finalHeaders = finalHeaders.Where(a => a != null).ToArray();
        foreach (var head in finalHeaders) csvWriter.WriteField(head);
        csvWriter.NextRecord();

        // Read rows and write selected fields
        while (csvReader.Read())
        {
            foreach (var head in finalHeaders)
            {
                var value = csvReader.GetField(head ?? throw new InvalidOperationException());
                csvWriter.WriteField(value);
            }
            csvWriter.NextRecord();
        }
    }

    public static void MergeCsvFiles(string[] inputFiles, string outputPath, string[] finalHeaders, bool addSourceFileColumn = false)
    {
        if (inputFiles == null || inputFiles.Length == 0)
            throw new ArgumentException("No input files provided to merge.");

        if (finalHeaders == null || finalHeaders.Length == 0)
            throw new ArgumentException("Provide at least one final header.");

        using var writer = new StreamWriter(outputPath);
        using var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // write canonical header
        foreach (var col in finalHeaders) csvWriter.WriteField(col);
        if (addSourceFileColumn) csvWriter.WriteField("source_file");
        csvWriter.NextRecord();

        foreach (var file in inputFiles.OrderBy(f => f))
        {
            if (!File.Exists(file)) continue;

            var fileName = Path.GetFileName(file);
            using var reader = new StreamReader(file);
            using var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture);

            if (!csvReader.Read()) continue;
            csvReader.ReadHeader();
            var header = csvReader.HeaderRecord ?? [];
            var mapping = finalHeaders.Select(col => header.Contains(col) ? col : null).ToArray();
            var missing = mapping.Where((c, i) => mapping[i] == null).ToArray();
            if (missing.Length != 0) throw new Exception($"Missing columns in input CSV: {string.Join(", ", missing)}");

            // stream rows
            mapping = mapping.Where(a => a != null).ToArray();
            while (csvReader.Read())
            {
                foreach (var col in mapping)
                {
                    string? val;
                    try { val = csvReader.GetField(col ?? throw new InvalidOperationException()); }
                    catch { val = string.Empty; }
                    csvWriter.WriteField(val);
                }

                if (addSourceFileColumn) csvWriter.WriteField(fileName);

                csvWriter.NextRecord();
            }
        }
    }

    public static void FilterCanceledOrders(
        string inputPath,
        string outputPath,
        string[]? defaults = null,
        string statusColumn = "Trạng Thái Đơn Hàng")
    {
        if (!File.Exists(inputPath)) throw new FileNotFoundException(inputPath);

        if (defaults == null || defaults.Length == 0) defaults = new[] { "Đã hủy" };
        var cancelSet = new HashSet<string>(defaults, StringComparer.OrdinalIgnoreCase);

        using var reader = new StreamReader(inputPath);
        using var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture);

        if (!csvReader.Read()) throw new Exception("Input CSV is empty.");
        csvReader.ReadHeader();
        var header = csvReader.HeaderRecord ?? [];
        var headerLookup = header.ToDictionary(h => h, h => h, StringComparer.OrdinalIgnoreCase);

        if (!header.Contains(statusColumn))
            throw new Exception($"Status column '{statusColumn}' not found in {inputPath}.");

        using var writer = new StreamWriter(outputPath);
        using var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // write header identical to input header
        foreach (var h in header) csvWriter.WriteField(h);
        csvWriter.NextRecord();

        long rowsIn = 0;
        long rowsOut = 0;
        long rowsFiltered = 0;
        while (csvReader.Read())
        {
            rowsIn++;
            string statusVal;
            try { statusVal = csvReader.GetField(statusColumn)?.Trim() ?? string.Empty; }
            catch { statusVal = string.Empty; }

            // If status is empty -> treat as not cancelled (keep). If you prefer to drop unknowns, change logic.
            var isCanceled = !string.IsNullOrEmpty(statusVal) && cancelSet.Contains(statusVal);

            if (isCanceled)
            {
                rowsFiltered++;
                continue;
            }

            // write all fields in the original header order
            foreach (var h in header)
            {
                string v;
                try { v = csvReader.GetField(h) ?? throw new InvalidOperationException(); } catch { v = string.Empty; }
                csvWriter.WriteField(v);
            }
            csvWriter.NextRecord();
            rowsOut++;
        }

        Console.WriteLine($"Filtered '{inputPath}' -> '{outputPath}'. In: {rowsIn}, Out: {rowsOut}, Removed: {rowsFiltered}.");
    }

    public static void LabelSkus(
        string inputPath,
        string outputPath,
        string productNameCol = "Tên sản phẩm",
        string variantNameCol = "Tên phân loại hàng",
        string productSkuCol = "SKU sản phẩm",
        string variantSkuCol = "SKU phân loại hàng")
    {
        if (!File.Exists(inputPath)) throw new ArgumentException("inputPath not found.");

        // Read whole CSV into list of dictionaries (safe for ~2k rows)
        var rows = new List<Dictionary<string, string>>();
        using var reader = new StreamReader(inputPath);
        using var csvReader = new CsvReader(reader, CultureInfo.InvariantCulture);

        if (!csvReader.Read()) throw new Exception("Empty input file");
        csvReader.ReadHeader();
        var header = csvReader.HeaderRecord ?? [];

        while (csvReader.Read())
        {
            var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var h in header) dict[h] = csvReader.GetField(h) ?? string.Empty;
            rows.Add(dict);
        }

        // Mapping structures
        var productNameToSku = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var productSkuToVariantMap = new Dictionary<string, Dictionary<string, string>>(StringComparer.OrdinalIgnoreCase);
        var productCounter = 0;
        var productVariantCounter = 0;

        // iterate rows and assign SKUs
        foreach (var r in rows)
        {
            r.TryGetValue(productNameCol, out var productNameRaw);
            r.TryGetValue(variantNameCol, out var variantNameRaw);

            var productName = (productNameRaw ?? string.Empty).Trim();
            var variantName = (variantNameRaw ?? string.Empty).Trim();

            // existing SKUs (if present)
            r.TryGetValue(productSkuCol, out var existingProductSku);
            r.TryGetValue(variantSkuCol, out var existingVariantSku);

            // Determine product SKU
            string productSku;
            if (!string.IsNullOrWhiteSpace(existingProductSku))
            {
                productSku = existingProductSku.Trim();
                // ensure mapping includes it
                if (!productNameToSku.ContainsKey(productName) && !string.IsNullOrEmpty(productName))
                    productNameToSku[productName] = productSku;
            }
            else
            {
                // try find mapping by product name
                if (!string.IsNullOrEmpty(productName) && productNameToSku.TryGetValue(productName, out var mapped))
                {
                    productSku = mapped;
                }
                else
                {
                    // create new product SKU
                    productSku = $"PRD{productCounter:D4}";
                    if (!string.IsNullOrEmpty(productName))
                        productNameToSku[productName] = productSku;
                    productCounter++;
                }
            }

            // Ensure variant map exists for this product SKU
            if (!productSkuToVariantMap.TryGetValue(productSku, out var variantMap))
            {
                variantMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                productSkuToVariantMap[productSku] = variantMap;
            }

            // Determine variant SKU
            string variantSku;
            if (!string.IsNullOrWhiteSpace(existingVariantSku))
            {
                variantSku = existingVariantSku.Trim();
                // add to variant map if name present
                if (!variantMap.ContainsKey(variantName) && !string.IsNullOrEmpty(variantName))
                    variantMap[variantName] = variantSku;
            }
            else
            {
                if (!string.IsNullOrEmpty(variantName))
                {
                    if (variantMap.TryGetValue(variantName, out var mappedVariant))
                    {
                        variantSku = mappedVariant;
                    }
                    else
                    {
                        // create new variant SKU for that product
                        var nextIdx = variantMap.Count + 1;
                        variantSku = $"{productSku}-V{nextIdx:D2}";
                        variantMap[variantName] = variantSku;
                        productVariantCounter++;
                    }
                }
                else
                {
                    // no variant name => use product SKU itself
                    variantSku = productSku;
                }
            }

            // write back into dictionary (add or overwrite)
            r[productSkuCol] = productSku;
            r[variantSkuCol] = variantSku;
        }

        // Write labeled CSV
        using (var writer = new StreamWriter(outputPath))
        using (var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            // write header (use headerList which may include added SKU columns)
            foreach (var h in header) csvWriter.WriteField(h);
            csvWriter.NextRecord();

            foreach (var r in rows)
            {
                foreach (var h in header)
                {
                    r.TryGetValue(h, out var v);
                    csvWriter.WriteField(v ?? string.Empty);
                }
                csvWriter.NextRecord();
            }
        }

        // print summary
        Console.WriteLine($"Labeling complete. Assigned {productCounter} new product SKUs. Assigned {productVariantCounter} new product variation SKUs.");
    }

    static void Main(string[] args)
    {
        try
        {
            var keep = new[]
            {
                "Mã đơn hàng",
                "Ngày đặt hàng",
                "SKU sản phẩm",
                "SKU phân loại hàng",
                "Tên sản phẩm",
                "Tên phân loại hàng",
                "Giá gốc",
                "Giá ưu đãi",
                "Số lượng",
                "Tổng giá bán (sản phẩm)",
                "Trạng Thái Đơn Hàng",
                "Số lượng sản phẩm được hoàn trả"
            };

            var inputFiles = Directory.EnumerateFiles(
                InputFolder,
                "*.csv",
                SearchOption.TopDirectoryOnly
            ).ToArray();
            var outputFiles = new List<string>();
            foreach (var input in inputFiles)
            {
                var output = input.Replace(@"Input\Order.all", @"Output\Filter");
                FilterCsvByColumns(input, output, keep, ignoreMissing: false);
                outputFiles.Add(output);
                Console.WriteLine($"Filtered CSV written to: {output}");
            }

            MergeCsvFiles(outputFiles.ToArray(), OutputFile, keep, addSourceFileColumn: false);
            Console.WriteLine($"Merged file written to: {OutputFile}");

            FilterCanceledOrders(OutputFile, FilterOutput);
            Console.WriteLine($"Filtered file written to: {FilterOutput}");

            LabelSkus(FilterOutput, LabeledOutput);
            Console.WriteLine($"Labeled file written to: {LabeledOutput}");

            OrderAggregator.AggregateToProductDaily(LabeledOutput, AggregatedOutput);
            Console.WriteLine($"Aggregated file written to: {AggregatedOutput}");

            DailyFeatureBuilder.BuildFeatures(AggregatedOutput, DailyBuildFeaturesOutput);
            Console.WriteLine($"Daily build features file written to: {DailyBuildFeaturesOutput}");

            WeeklyFeatureBuilder.BuildFeatures(AggregatedOutput, WeeklyBuildFeaturesOutput);
            Console.WriteLine($"weekly build features file written to: {WeeklyBuildFeaturesOutput}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}