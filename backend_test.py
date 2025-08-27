#!/usr/bin/env python3
"""
Lead Scoring API Testing Suite
Tests both FastAPI (port 5051) and Flask (port 5052) services
"""

import requests
import json
import sys
from datetime import datetime
import time

class LeadScoringAPITester:
    def __init__(self):
        self.fastapi_base_url = "http://127.0.0.1:5051"
        self.flask_base_url = "http://127.0.0.1:5052"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, test_name, success, details):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
        
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        print()

    def test_fastapi_health(self):
        """Test FastAPI health endpoint"""
        try:
            response = requests.get(f"{self.fastapi_base_url}/v2/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['status', 'model_loaded', 'preprocessor_loaded', 'predictions_count']
                
                # Check required fields
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    self.log_test("FastAPI Health Check", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check model and preprocessor loaded
                if data.get('model_loaded') and data.get('preprocessor_loaded'):
                    status = data.get('status', 'unknown')
                    self.log_test("FastAPI Health Check", True, 
                                f"Status: {status}, Model: {data['model_loaded']}, Preprocessor: {data['preprocessor_loaded']}, Predictions: {data['predictions_count']}")
                    return True, data['predictions_count']
                else:
                    self.log_test("FastAPI Health Check", False, 
                                f"Models not loaded - Model: {data.get('model_loaded')}, Preprocessor: {data.get('preprocessor_loaded')}")
                    return False
            else:
                self.log_test("FastAPI Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("FastAPI Health Check", False, f"Exception: {str(e)}")
            return False

    def test_fastapi_predict(self, test_name="FastAPI Predict"):
        """Test FastAPI prediction endpoint"""
        payload = {
            "TotalVisits": 5,
            "Page Views Per Visit": 3.2,
            "Total Time Spent on Website": 1850,
            "Lead Origin": "API",
            "Lead Source": "Google",
            "Last Activity": "Email Opened",
            "What is your current occupation": "Working Professional"
        }
        
        try:
            response = requests.post(
                f"{self.fastapi_base_url}/v2/predict",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['prediction', 'lead_score', 'label', 'timestamp', 'model_version']
                
                # Check required fields
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    self.log_test(test_name, False, f"Missing fields: {missing_fields}")
                    return False
                
                # Validate field types
                if (isinstance(data.get('prediction'), int) and 
                    isinstance(data.get('lead_score'), (int, float)) and
                    isinstance(data.get('label'), str)):
                    
                    self.log_test(test_name, True, 
                                f"Prediction: {data['prediction']}, Score: {data['lead_score']}%, Label: {data['label']}")
                    return True
                else:
                    self.log_test(test_name, False, f"Invalid field types in response: {data}")
                    return False
            else:
                self.log_test(test_name, False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")
            return False

    def test_fastapi_models_info(self):
        """Test FastAPI models info endpoint"""
        try:
            response = requests.get(f"{self.fastapi_base_url}/v2/models/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for expected fields
                if ('expected_features' in data and 
                    'model_loaded' in data and 
                    'preprocessor_loaded' in data):
                    
                    features_count = len(data.get('expected_features', []))
                    self.log_test("FastAPI Models Info", True, 
                                f"Features count: {features_count}, Model loaded: {data['model_loaded']}, Preprocessor loaded: {data['preprocessor_loaded']}")
                    return True
                else:
                    self.log_test("FastAPI Models Info", False, f"Missing expected fields in response: {data}")
                    return False
            else:
                self.log_test("FastAPI Models Info", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("FastAPI Models Info", False, f"Exception: {str(e)}")
            return False

    def test_flask_health(self):
        """Test Flask health endpoint"""
        try:
            response = requests.get(f"{self.flask_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                if ('model_loaded' in data and 
                    'preprocessor_loaded' in data and 
                    'status' in data):
                    
                    if data.get('model_loaded') and data.get('preprocessor_loaded'):
                        self.log_test("Flask Health Check", True, 
                                    f"Status: {data['status']}, Model: {data['model_loaded']}, Preprocessor: {data['preprocessor_loaded']}")
                        return True
                    else:
                        self.log_test("Flask Health Check", False, 
                                    f"Models not loaded - Model: {data.get('model_loaded')}, Preprocessor: {data.get('preprocessor_loaded')}")
                        return False
                else:
                    self.log_test("Flask Health Check", False, f"Missing expected fields in response: {data}")
                    return False
            else:
                self.log_test("Flask Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Flask Health Check", False, f"Exception: {str(e)}")
            return False

    def test_prediction_count_increment(self):
        """Test that prediction count increments after multiple predictions"""
        print("🔍 Testing prediction count increment...")
        
        # Get initial count
        initial_health = self.test_fastapi_health()
        if not initial_health:
            self.log_test("Prediction Count Increment", False, "Could not get initial health status")
            return False
        
        initial_count = initial_health[1] if isinstance(initial_health, tuple) else 0
        
        # Make two predictions
        pred1 = self.test_fastapi_predict("FastAPI Predict #1 (for count test)")
        pred2 = self.test_fastapi_predict("FastAPI Predict #2 (for count test)")
        
        if not (pred1 and pred2):
            self.log_test("Prediction Count Increment", False, "Predictions failed")
            return False
        
        # Check final count
        time.sleep(1)  # Give background task time to complete
        final_health = self.test_fastapi_health()
        if not final_health:
            self.log_test("Prediction Count Increment", False, "Could not get final health status")
            return False
        
        final_count = final_health[1] if isinstance(final_health, tuple) else 0
        
        if final_count >= initial_count + 2:
            self.log_test("Prediction Count Increment", True, 
                        f"Count increased from {initial_count} to {final_count}")
            return True
        else:
            self.log_test("Prediction Count Increment", False, 
                        f"Count did not increment properly: {initial_count} -> {final_count}")
            return False

    def test_minimal_payload(self):
        """Test FastAPI with minimal required fields"""
        minimal_payload = {
            "TotalVisits": 1,
            "Page Views Per Visit": 1.0,
            "Total Time Spent on Website": 100,
            "Lead Origin": "API",
            "Lead Source": "Google",
            "Last Activity": "Email Opened",
            "What is your current occupation": "Student"
        }
        
        try:
            response = requests.post(
                f"{self.fastapi_base_url}/v2/predict",
                json=minimal_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("FastAPI Minimal Payload", True, 
                            f"Prediction: {data.get('prediction')}, Score: {data.get('lead_score')}%")
                return True
            else:
                self.log_test("FastAPI Minimal Payload", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("FastAPI Minimal Payload", False, f"Exception: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Lead Scoring API Tests")
        print("=" * 50)
        
        # Test FastAPI endpoints
        print("📡 Testing FastAPI Service (port 5051)")
        print("-" * 30)
        self.test_fastapi_health()
        self.test_fastapi_predict()
        self.test_fastapi_models_info()
        self.test_minimal_payload()
        
        # Test Flask endpoints
        print("🌶️  Testing Flask Service (port 5052)")
        print("-" * 30)
        self.test_flask_health()
        
        # Test edge cases
        print("🔬 Testing Edge Cases")
        print("-" * 30)
        self.test_prediction_count_increment()
        
        # Print summary
        print("📊 Test Summary")
        print("=" * 50)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed. Check logs above for details.")
            return 1

def main():
    """Main test runner"""
    tester = LeadScoringAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())