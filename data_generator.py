import io
import os
import tempfile
import hashlib
import numpy as np
import pandas as pd
import concurrent.futures

class DataGenerator:
    def __init__(self, num_rows=10000):
        self.num_rows = num_rows
        self.df = self._generate_data()
    
    def _generate_data(self):
        """Generates a DataFrame with varied data types."""
        # id and hash_id
        ids = np.arange(1, self.num_rows + 1)
        hash_ids = [hashlib.sha256(str(x).encode('utf-8')).hexdigest() for x in ids]

        # Generate names using predefined first and last names
        first_names = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank'])
        last_names = np.array(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia'])
        names = np.core.defchararray.add(
            np.random.choice(first_names, self.num_rows),
            np.core.defchararray.add(" ", np.random.choice(last_names, self.num_rows))
        )

        # Generate addresses (house number, street, city)
        streets = np.array(['Main St', 'Oak St', 'Pine St', 'Maple Ave', 'Cedar Ave', 'Elm St'])
        cities = np.array(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
        house_numbers = np.random.randint(1, 10000, size=self.num_rows).astype(str)
        addresses = [
            f"{num} {street}, {city}" 
            for num, street, city in zip(
                house_numbers, 
                np.random.choice(streets, self.num_rows), 
                np.random.choice(cities, self.num_rows)
            )
        ]
        # Random datetime between 2000 and 2025
        start_ts = pd.Timestamp('2000-01-01').value // 10**9
        end_ts = pd.Timestamp('2025-12-31 23:59:59').value // 10**9
        random_ts = np.random.randint(start_ts, end_ts, size=self.num_rows)
        datetimes = pd.to_datetime(random_ts, unit='s')
        dates = datetimes.date

        # Numeric fields
        int_field = np.random.randint(0, 10000, size=self.num_rows)
        float_field = np.random.uniform(0, 1000, size=self.num_rows)

        # Text fields
        words = np.array(['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit'])
        sentence_length = 5
        rand_indices = np.random.randint(0, len(words), size=(self.num_rows, sentence_length))
        text_field = [' '.join(words[row]) for row in rand_indices]
        string_field = np.random.choice(words, self.num_rows)

        data = {
            'id': ids,
            'hash_id': hash_ids,
            'name': names,
            'address': addresses,
            'datetime': datetimes,
            'date': dates,
            'int_field': int_field,
            'float_field': float_field,
            'text_field': text_field,
            'string_field': string_field
        }
        return pd.DataFrame(data)
    
    # ---------------
    # Methods returning file paths
    # ---------------
    def to_csv_file(self):
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.df.to_csv(temp_csv.name, index=False)
        return temp_csv.name

    def to_json_file(self):
        temp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.df.to_json(temp_json.name, orient='records', date_format='iso')
        return temp_json.name

    def to_excel_file(self):
        temp_excel = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        with pd.ExcelWriter(temp_excel.name, engine='openpyxl') as writer:
            self.df.to_excel(writer, index=False)
        return temp_excel.name

    def to_parquet_file(self):
        temp_parquet = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        self.df.to_parquet(temp_parquet.name)
        return temp_parquet.name
    
    # ---------------
    # Methods returning in-memory streams (io.BytesIO)
    # ---------------
    def to_csv_stream(self):
        stream = io.BytesIO()
        self.df.to_csv(stream, index=False)
        stream.seek(0)
        return stream

    def to_json_stream(self):
        stream = io.BytesIO()
        self.df.to_json(stream, orient='records', date_format='iso')
        stream.seek(0)
        return stream

    def to_excel_stream(self):
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            self.df.to_excel(writer, index=False)
        stream.seek(0)
        return stream

    def to_parquet_stream(self):
        stream = io.BytesIO()
        self.df.to_parquet(stream)
        stream.seek(0)
        return stream

    # ---------------
    # Utility method to generate all outputs concurrently
    # ---------------
    def generate_all_files(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                'csv': executor.submit(self.to_csv_file),
                'json': executor.submit(self.to_json_file),
                'excel': executor.submit(self.to_excel_file),
                'parquet': executor.submit(self.to_parquet_file),
            }
            return {fmt: future.result() for fmt, future in futures.items()}

    def generate_all_streams(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                'csv': executor.submit(self.to_csv_stream),
                'json': executor.submit(self.to_json_stream),
                'excel': executor.submit(self.to_excel_stream),
                'parquet': executor.submit(self.to_parquet_stream),
            }
            return {fmt: future.result() for fmt, future in futures.items()}

# -----------------------
# Demonstration
# -----------------------
def main():
    # Initialize data generator with 10,000 rows (adjust as needed)
    generator = DataGenerator(num_rows=10000)
    
    # Generate files concurrently and print file paths
    file_paths = generator.generate_all_files()
    print("Generated file paths:")
    for fmt, path in file_paths.items():
        print(f"{fmt.upper()}:", path)

    # Generate in-memory streams concurrently and print stream sizes
    streams = generator.generate_all_streams()
    print("\nIn-memory stream sizes (in bytes):")
    for fmt, stream in streams.items():
        print(f"{fmt.upper()} stream:", len(stream.getvalue()))

    # Optionally, remove temporary files after use
    for path in file_paths.values():
        os.unlink(path)

if __name__ == "__main__":
    main()
