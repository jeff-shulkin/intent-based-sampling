'''
Definition for Intent class.
This class takes in an arbitrary number of priorities, 
'''
import sys
from typing import Dict


class Intent():
    # interpretation methods
    # TODO: Figure out what methods to evaluate. Linear and ML are a good start, not sure about MPC.
    m_intent_method_options = ['linear', 'mpc', 'ml']

    # member variables
    m_intent_priorities = {}
    m_intent_method = ''


    # Assume that priorities are static relative to the downstream application
    def __init__(self, priority_vector: Dict[str, float], method='linear'):
        if not self.validate_priority_vector(priority_vector):
            sys.exit(1)

        if not self.validate_method(method):
            sys.exit(1)
        
        # Now that we know our arguments are properly formed, pass them to the internal class variables
        self.m_intent_priorities = priority_vector
        self.m_intent_method = method

    def update_priorities(self, priority_vector: Dict[str, float]):
        if not self.validate_priority_vector(priority_vector):
            sys.exit(1)

        self.m_intent_priorities = priority_vector

    # Validate Priority vector. Return false if the vector is malformed.
    def validate_priority_vector(self, priority_vector: Dict[str, float]) -> bool:
        valid = False
        try:
            for p_name, p_val in priority_vector.items():
                # Check that that each name is paired with an int/float type
                if not isinstance(p_val, (int, float)):
                    raise TypeError(f"{p_name} priority must be a number, got {type(p_val)}")
            
                # Check that each priority value is between 0 and 1
                if not (0.0 <= p_val <= 1.0):
                    raise ValueError(f"{p_name} priority must be a number between 0.0 and 1.0, got {p_val}")
                
                valid = True
        
        except (TypeError, ValueError) as e:
            print(f"Priority Vector malformed with exception {e}. Terminating application.")

        return valid

    def validate_method(self, method: str) -> bool:
        return method in self.m_intent_method_options