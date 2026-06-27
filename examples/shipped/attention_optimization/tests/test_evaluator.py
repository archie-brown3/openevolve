#!/usr/bin/env python3
"""
Test script to verify the complete AlphaEvolve setup works
"""

import sys
import json
from pathlib import Path

def test_evaluator():
    """Test the evaluator with a simple program"""
    
    print("🧪 Testing evaluator...")
    
    # Simple test program
    test_program = '''
def optimize_attention():
    return {
        'tile_size_m': 32,
        'tile_size_n': 64,
        'vectorization': 'none',
        'unroll_factor': 2,
        'loop_interchange': False,
        'fusion_strategy': 'none',
        'use_shared_memory': False,
        'optimize_for_latency': True,
        'enable_blocking': False,
        'enable_recomputation': False,
        'optimization_strategy': 'alphaevolve_test',
        'target_speedup': 1.32,
    }
'''
    
    try:
        # Import the evaluator
        sys.path.insert(0, '.')
        from examples.shipped.attention_optimization.evaluator import evaluate_program
        
        print("✅ Evaluator imported successfully")
        
        # Test evaluation
        result = evaluate_program(test_program)
        
        if 'error' in result:
            print(f"📊 Evaluation result: error={result['error']:.3f}")
            if 'speedup' in result:
                print(f"📊 Speedup: {result['speedup']:.3f}x")
            if 'mlir_source' in result:
                print(f"📂 MLIR source: {result['mlir_source']}")
                
            if result['error'] < 1000:
                print("✅ Evaluator works!")
                return True
            else:
                print(f"❌ Evaluator failed: {result}")
                return False
        else:
            print(f"❌ Invalid result format: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluator test failed: {e}")
        return False

def test_initial_program():
    """Test the initial program generates parameters"""
    
    print("\n🧪 Testing initial program...")
    
    try:
        sys.path.insert(0, '.')
        from examples.shipped.attention_optimization.initial_program import optimize_attention
        
        params = optimize_attention()
        
        print("✅ Initial program imported successfully")
        print(f"📊 Generated parameters: {list(params.keys())}")
        
        # Check required parameters
        required = ['tile_size_m', 'tile_size_n', 'unroll_factor']
        for param in required:
            if param in params:
                print(f"✅ {param}: {params[param]}")
            else:
                print(f"❌ Missing parameter: {param}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Initial program test failed: {e}")
        return False

def test_mlir_file():
    """Test that the MLIR file exists and is readable"""
    
    print("\n🧪 Testing MLIR file...")
    
    mlir_file = Path("./mlir/self_attn_with_consts_linalg_dialect.mlir")
    
    if mlir_file.exists():
        print(f"✅ MLIR file exists: {mlir_file}")
        try:
            with open(mlir_file, 'r') as f:
                content = f.read()
                print(f"✅ MLIR file readable: {len(content)} characters")
                
                # Check for fixed tensor.expand_shape syntax
                if 'output_shape' in content:
                    print("✅ tensor.expand_shape syntax is fixed")
                else:
                    print("⚠️ tensor.expand_shape may need fixing")
                
                return True
        except Exception as e:
            print(f"❌ Cannot read MLIR file: {e}")
            return False
    else:
        print(f"❌ MLIR file not found: {mlir_file}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Testing Complete AlphaEvolve Setup\n")
    
    tests = [
        ("MLIR File", test_mlir_file),
        ("Initial Program", test_initial_program), 
        ("Evaluator", test_evaluator),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to run AlphaEvolve!")
        print("\n🚀 Run evolution with:")
        print("   python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 10")
        print("\n🎯 Target: Achieve 32% speedup (1.32x) like AlphaEvolve paper")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Fix issues before running evolution.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)