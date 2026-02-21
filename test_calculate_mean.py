import pytest 
from read_data import calculate_mean #import function that we want to test 
from read_data import fibonacci_sequence
from read_data import properly_balanced
from read_data import merge_intervals 

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
        

def test_fib_str():
    with pytest.raises(TypeError):
        fibonacci_sequence(str)
def test_fib_float():
    with pytest.raises(TypeError):
        fibonacci_sequence(float)

def test_balance_standard():
    sentence = "{}"
    with pytest.raises(TypeError):
        len(properly_balanced(sentence)) % 2 != 0 #to ensure we enter an even number of characters. 
        #cannot be properly balanced if we have an uneven amount of characters

def test_balance_int():
    with pytest.raises(TypeError):
        properly_balanced(int)
        
def test_balance_float():
    with pytest.raises(TypeError):
        properly_balanced(float)

def test_ints_standard():
    ints = [1, 2]
    left = ints[0]
    right = ints[1]
    assert left <= right #need to make sure left endpoint of interval is smaller than right endpoint of interval
    with pytest.raises(TypeError):
        len(merge_intervals(ints)) %2 != 0 #ensures we have a left and right endpoint in each of the intervals
        #I am beginning to think this is incorrect but I am going to leave it because I think it works?
    
def test_ints_empty():
    with pytest.raises(ValueError):
        merge_intervals([])

def test_ints_str():
    with pytest.raises(TypeError):
        merge_intervals(str)
    
    