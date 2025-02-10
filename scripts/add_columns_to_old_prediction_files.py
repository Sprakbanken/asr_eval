import pandas as pd
from pathlib import Path


def add_columns_to_old_df(old_df_p: Path, new_df: pd.DataFrame, merge_col: str):
    print(old_df_p)
    old_df = pd.read_csv(old_df_p)
    if len(old_df) != len(new_df):
        print("Dataframes must have the same number of rows\n")
        return

    assert old_df[merge_col].is_unique, f"{merge_col} column must be unique"

    merged_df = old_df.merge(
        new_df, on=merge_col, how="inner", suffixes=("_2024", "_2023")
    )

    assert len(merged_df) == len(old_df) and len(merged_df) == len(new_df)

    overlapping_columns = [
        col for col in old_df.columns if col in new_df.columns and col != merge_col
    ]

    # Check if the values in the overlapping columns are the same for each row
    for col in overlapping_columns:
        if not merged_df[f"{col}_2024"].equals(merged_df[f"{col}_2023"]):
            print(f"\tValues in column '{col}' do not match between the dataframes.")
            print(
                f"\tNum rows with different values {len(merged_df[merged_df[f'{col}_2024'] != merged_df[f'{col}_2023']])}\n"
            )
        else:
            merged_df[col] = merged_df[f"{col}_2024"]
            merged_df.drop(columns=[f"{col}_2024", f"{col}_2023"], inplace=True)

    with open(old_df_p, "w") as f:
        merged_df.to_csv(f, index=False)


if __name__ == "__main__":
    # # # Change as needed # # #
    new_df_p = Path(
        "data/output/2024/2024-12-23_NbAiLab_nb-whisper-large-distil-turbo-beta_nn.csv"
    )
    preds_2023_p = Path("data/output/2023/")
    merge_col = "segmented_audio"
    # # # # # # # # # # # # # #

    new_df = pd.read_csv(new_df_p)
    new_df = new_df.drop(columns=["predictions"])
    assert new_df[merge_col].is_unique, f"{merge_col} column must be unique"

    for e in preds_2023_p.glob("**/*.csv"):
        add_columns_to_old_df(e, new_df=new_df, merge_col="segmented_audio")
