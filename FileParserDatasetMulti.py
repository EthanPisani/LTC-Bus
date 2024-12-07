import json
import csv
import pandas as pd
import glob
import time
import tqdm
import multiprocessing
from multiprocessing import Pool

# Function to process files in parallel
def process_files(file_names):
    extracted_data = []
    seen_stops = set()
    stop_id_map = {}

    for file_name in file_names:
        time_stamp = int(file_name.split("TripUpdate")[1].split(".json")[0])

        # Read and process the JSON file
        with open(file_name, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading file: {file_name}")
                continue

            for entity in data.get("entity", []):
                entity_id = entity.get("id", None)
                trip_update = entity.get("trip_update", {})
                vehicle_info = trip_update.get("vehicle", {})
                vehicle_id = vehicle_info.get("id", None) if vehicle_info else None

                stop_time_updates = trip_update.get("stop_time_update", [])
                trip = trip_update.get("trip", {})
                route_id = trip.get("route_id", "")

                for stop in stop_time_updates:
                    arrival_info = stop.get("arrival", {})
                    departure_info = stop.get("departure", {})

                    stop_id = stop.get("stop_id", None)

                    if arrival_info:
                        delay = arrival_info.get("delay", None)
                        scheduled_time = arrival_info.get("schedule_time", None)
                        actual_time = arrival_info.get("time", None)
                    elif departure_info:
                        delay = departure_info.get("delay", None)
                        scheduled_time = departure_info.get("schedule_time", None)
                        actual_time = departure_info.get("time", None)
                    else:
                        delay = None
                        scheduled_time = None
                        actual_time = None

                    if actual_time and time_stamp - 60 <= actual_time < time_stamp + 60:
                        # Efficiently convert times
                        ts = pd.to_datetime(scheduled_time, unit='s')
                        scheduled_time_second = ts.hour * 3600 + ts.minute * 60 + ts.second
                        ts2 = pd.to_datetime(actual_time, unit='s')
                        actual_time_second = ts2.hour * 3600 + ts2.minute * 60 + ts2.second

                        day_of_week = ts.weekday()
                        day_of_year = ts.dayofyear
                        if stop_id not in seen_stops:
                            seen_stops.add(stop_id)
                            stop_id_map[stop_id] = len(stop_id_map)

                        extracted_data.append({
                            "route_id": route_id,
                            "vehicle_id": vehicle_id,
                            "stop_id": stop_id_map[stop_id],
                            "delay": delay,
                            "scheduled_time": scheduled_time_second,
                            "actual_time": actual_time_second,
                            "day": day_of_week,
                            "day_of_year": day_of_year
                        })

    return extracted_data

# Function to split data into 15 CSVs and save
def save_data_to_csv(extracted_data):
    # Determine the number of parts (15 parts)
    num_parts = 18
    chunk_size = len(extracted_data) // num_parts

    for i in range(num_parts):
        # Determine the slice of data for this part
        start_index = i * chunk_size
        # Ensure the last part gets any remaining data
        end_index = (i + 1) * chunk_size if i < num_parts - 1 else len(extracted_data)

        chunk_data = extracted_data[start_index:end_index]

        # Writing the chunk to a CSV file
        csv_file = f"./extracted2/extracted_data_part_{i + 1}.csv"
        fieldnames = ["route_id", "vehicle_id", "stop_id", "delay", "scheduled_time", "actual_time", "day", "day_of_year"]

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk_data)

        print(f"Data has been written to {csv_file}")

# Main function
def main():
    # Glob pattern to get all relevant files
    file_pattern = '/home/stephen/raw_dataset/londonBusData/TripUpdate*.json'
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files to process")

    # Reduce the number of files to process (for example, limit to 10000 files)
    # files = files[:10000]

    # Split files into 5 roughly equal parts for multiprocessing
    num_processes = 10
    chunk_size = len(files) // num_processes
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    print(f"Processing {len(file_chunks)} chunks of files")
    print(f"Each chunk contains {chunk_size} files")
    # Use multiprocessing to process files in parallel
    with Pool(processes=num_processes) as pool:
        # Process each chunk of files in parallel
        results = pool.map(process_files, file_chunks)

    # Flatten the list of results into a single list
    extracted_data = [item for sublist in results for item in sublist]

    # Save the extracted data to CSVs
    save_data_to_csv(extracted_data)

    print("Total processing complete.")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Total time taken: 9139.18 seconds   
# about 2.5 hours