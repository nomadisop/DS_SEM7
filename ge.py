import pandas as pd
from great_expectations.dataset import PandasDataset
from great_expectations.data_context import DataContext
import great_expectations as ge

# Load the dataset (for example, a CSV file)
df = pd.read_csv('cleaned_featured_books_dataset.py')

# Convert the pandas DataFrame to a Great Expectations dataset
ge_df = PandasDataset(df)

# Step 1: Define Expectations
# Example: Expect column 'age' to be greater than 18
ge_df.expect_column_values_to_be_greater_than('age', 18)

# Example: Expect column 'name' to contain non-null values
ge_df.expect_column_values_to_be_in_set('name', ge_df['name'].dropna().unique())

# Example: Expect column 'age' to be of type integer
ge_df.expect_column_values_to_be_in_type_list('age', ['int64'])

# Step 2: Validate the Dataset
validation_result = ge_df.validate()

# Step 3: Print the Validation Results
print("Validation Results:")
print(validation_result)

# Step 4: Create a DataContext to save the validation to an Expectation Suite (Optional)
context = DataContext()

# Create an Expectation Suite if not exists
suite_name = "my_suite"
suite = context.create_expectation_suite(suite_name, overwrite_existing=True)

# Save the expectations into the suite
ge_df.save_expectation_suite(suite_name)

# Step 5: Run the validation with the context (optional for advanced use cases)
results = context.run_validation_operator(
    "action_list_operator", 
    validation_result
)

# Print the results from the context (Advanced usage)
print("Validation Results with DataContext:")
print(results)
