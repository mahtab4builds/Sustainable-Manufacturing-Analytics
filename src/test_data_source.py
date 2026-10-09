from library.data_source import load_data


def main():
    print("=" * 72)
    print("DATA SOURCE LIBRARY TEST")
    print("=" * 72)

    try:
        df = load_data()

        print()
        print("DATA SOURCE TEST PASSED")
        print("-" * 72)
        print(f"Rows    : {len(df)}")
        print(f"Columns : {len(df.columns)}")
        print()
        print("Columns detected:")
        print(", ".join(df.columns.astype(str)))
        print("=" * 72)

    except Exception as exc:
        print()
        print("=" * 72)
        print("DATA SOURCE TEST FAILED")
        print("=" * 72)
        print(f"Reason: {exc}")
        print("=" * 72)


if __name__ == "__main__":
    main()