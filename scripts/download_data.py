from dotenv import load_dotenv
import os
import databento as db

def main():
    load_dotenv()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    datasets = client.metadata.list_datasets()
    print(datasets)

    result = client.symbology.resolve(dataset="GLBX.MDP3", 
                            symbols=["MNQ.c.0"], 
                            stype_in="continuous", 
                            stype_out="instrument_id", 
                            start_date="2025-09-02", 
                            end_date="2025-09-03")
    print(result)


if __name__ == "__main__":
    main()