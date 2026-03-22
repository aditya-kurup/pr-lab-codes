import pandas as pd
import re
import seaborn as sns

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

while True:
    # Get column input
    print("Available columns: 'name', 'age', 'sex', 'survived', 'pclass'")
    column = input("Enter column to search: ").lower()

    if column not in ['name', 'age', 'sex', 'survived', 'pclass']:
        print("Invalid column, try again.")
        continue

    # Get pattern input
    pattern = input(f"Enter pattern to match in '{column}': ")

    # Match and print results
    results = [val for val in titanic[column] if re.match(pattern, str(val))]

    if results:
        print(f"\nFound {len(results)} matches:")
        for r in results:
            print(r)
    else:
        print("No matches found.")
        
    # Ask to search again
    again = input("\nSearch again? (yes/no): ")
    again = again.strip()
    again = again.lower()

    if again != 'yes':
        print("Goodbye!")
        break