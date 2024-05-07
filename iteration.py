# Initialize variables
current = 1
previous = 0

# Iterate through the first 10 numbers
for i in range(1, 11):
    # Calculate the sum of the current and previous number
    sum = current + previous
    
    # Print the current and previous number
    print(f"Previous number: {previous}, Current number: {current}")
    
    # Update the variables for the next iteration
    previous = current
    current = sum
