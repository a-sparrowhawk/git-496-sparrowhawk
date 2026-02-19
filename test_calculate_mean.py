import pytest 
from read_data import calculate_mean #import function that we want to test 
from read_data import fibonacci_sequence
from read_data import properly_balanced

#this is for the standard case when we have a list of numbers 
def test_mean_standard():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    #we assert (we ensure/state) that the mean of this list is above 5.5
    #if function is implemented correctly, it will pass this test. 
    #if the function is not implemented correctly, it will show is the error 
    assert calculate_mean(numbers) == 5.5
    
#this is for the edge case when the list is empty
def test_mean_empty():
    # if the list is empty, our code should raise an error (called an exception)
    with pytest.raises(ValueError):
        calculate_mean([])

def test_mean_string():
    with pytest.raises(TypeError):
        calculate_mean(str)
        
def test_fib_sequence():
    num = [0, 1]
    assert len(fibonacci_sequence(num)) >= 2
    with pytest.raises(TypeError):
        fibonacci_sequence(str)
    with pytest.raises(TypeError):
        fibonacci_sequence(float)

def test_properly_balance():
    sentence = "{}"
    with pytest.raises(TypeError):
        properly_balanced(int)
    with pytest.raises(TypeError):
        properly_balanced(float)
    assert len(properly_balanced(sentence)) % 2 == 0 #to ensure we enter an even number of characters. 
    #cannot be properly balanced if we have an uneven amount of characters
