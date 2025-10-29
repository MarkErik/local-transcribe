#!/usr/bin/env python3
"""
Comprehensive test runner for all cross-talk functionality tests.
This script runs all cross-talk related tests and provides a summary report.
"""

import sys
from pathlib import Path
import unittest
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all test modules
from test_basic_cross_talk import TestBasicCrossTalk
from test_integration_cross_talk import TestIntegrationCrossTalk
from test_txt_writer_cross_talk import TestTxtWriterCrossTalk
from test_csv_writer_cross_talk import TestCsvWriterCrossTalk
from test_turns_cross_talk import TestTurnsCrossTalk
from test_cli_cross_talk import TestCliCrossTalk
from test_end_to_end_cross_talk import TestEndToEndCrossTalk
from test_error_handling_cross_talk import TestErrorHandlingCrossTalk
from test_performance_cross_talk import TestPerformanceCrossTalk


def run_all_tests():
    """Run all cross-talk tests and provide a comprehensive report."""
    print("Running comprehensive cross-talk functionality tests...")
    print("=" * 80)
    
    # Create a test suite with all test classes
    test_classes = [
        TestBasicCrossTalk,
        TestIntegrationCrossTalk,
        TestTxtWriterCrossTalk,
        TestCsvWriterCrossTalk,
        TestTurnsCrossTalk,
        TestCliCrossTalk,
        TestEndToEndCrossTalk,
        TestErrorHandlingCrossTalk,
        TestPerformanceCrossTalk
    ]
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests with a custom runner that captures results
    start_time = time.time()
    
    class CustomTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
        
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append((test, 'SUCCESS'))
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append((test, 'FAILURE'))
        
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append((test, 'ERROR'))
    
    # Create a test runner with our custom result class
    runner = unittest.TextTestRunner(verbosity=2, resultclass=CustomTestResult)
    result = runner.run(suite)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CROSS-TALK TEST REPORT")
    print("=" * 80)
    
    # Overall summary
    print(f"\nOVERALL SUMMARY:")
    print(f"  - Tests run: {result.testsRun}")
    print(f"  - Failures: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    print(f"  - Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"  - Total time: {total_time:.2f} seconds")
    
    # Summary by test class
    print(f"\nSUMMARY BY TEST CLASS:")
    
    if hasattr(result, 'test_results'):
        # Group results by test class
        class_results = {}
        for test, status in result.test_results:
            class_name = test.__class__.__name__
            if class_name not in class_results:
                class_results[class_name] = {'SUCCESS': 0, 'FAILURE': 0, 'ERROR': 0}
            class_results[class_name][status] += 1
        
        # Print results for each class
        for class_name, counts in class_results.items():
            total = sum(counts.values())
            success_rate = (counts['SUCCESS'] / total * 100) if total > 0 else 0
            status_icon = "✅" if counts['FAILURE'] == 0 and counts['ERROR'] == 0 else "❌"
            print(f"  {status_icon} {class_name}: {counts['SUCCESS']}/{total} passed ({success_rate:.1f}%)")
    
    # Detailed failure information
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"  {i}. {test}:")
            # Print just the first line of the traceback
            first_line = traceback.split('\n')[1].strip()
            print(f"     {first_line}")
    
    # Detailed error information
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"  {i}. {test}:")
            # Print just the first line of the traceback
            first_line = traceback.split('\n')[1].strip()
            print(f"     {first_line}")
    
    # Test coverage summary
    print(f"\nTEST COVERAGE SUMMARY:")
    coverage_areas = [
        ("Basic cross-talk detection", TestBasicCrossTalk),
        ("Integration with diarization", TestIntegrationCrossTalk),
        ("Text output formatting", TestTxtWriterCrossTalk),
        ("CSV output formatting", TestCsvWriterCrossTalk),
        ("Turn building with cross-talk", TestTurnsCrossTalk),
        ("CLI integration", TestCliCrossTalk),
        ("End-to-end pipeline", TestEndToEndCrossTalk),
        ("Error handling", TestErrorHandlingCrossTalk),
        ("Performance testing", TestPerformanceCrossTalk)
    ]
    
    for area_name, test_class in coverage_areas:
        # Count tests for this class
        test_count = len(loader.loadTestsFromTestCase(test_class)._tests)
        print(f"  - {area_name}: {test_count} tests")
    
    # Final verdict
    print(f"\nFINAL VERDICT:")
    if result.wasSuccessful():
        print("  ✅ ALL TESTS PASSED!")
        print("     Cross-talk functionality is working correctly across all components.")
    else:
        print("  ❌ SOME TESTS FAILED!")
        print("     Please review the failures and errors above.")
    
    print("\n" + "=" * 80)
    
    return result.wasSuccessful()


def run_specific_test_class(test_class_name):
    """Run tests for a specific test class."""
    # Map of class names to actual classes
    class_map = {
        'basic': TestBasicCrossTalk,
        'integration': TestIntegrationCrossTalk,
        'txt_writer': TestTxtWriterCrossTalk,
        'csv_writer': TestCsvWriterCrossTalk,
        'turns': TestTurnsCrossTalk,
        'cli': TestCliCrossTalk,
        'end_to_end': TestEndToEndCrossTalk,
        'error_handling': TestErrorHandlingCrossTalk,
        'performance': TestPerformanceCrossTalk
    }
    
    if test_class_name not in class_map:
        print(f"Unknown test class: {test_class_name}")
        print(f"Available classes: {list(class_map.keys())}")
        return False
    
    test_class = class_map[test_class_name]
    
    print(f"Running {test_class.__name__} tests...")
    print("=" * 60)
    
    # Create and run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Check if a specific test class was requested
    if len(sys.argv) > 1:
        test_class_name = sys.argv[1].lower()
        success = run_specific_test_class(test_class_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)