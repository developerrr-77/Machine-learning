from utils import read_file, clean_data, transform_data

if __name__ == "__main__":
    df = read_file("SundasPoDetail.xlsx")
    df = clean_data(df)
    df = transform_data(df, "SundasPoDetail_transformed.xlsx")
