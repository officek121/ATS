import os
import subprocess
import sys

def main():
    print("Setting up Zero-Cost ATS Prototype Environment...")
    
    # Check if optimum-cli is available
    try:
        import optimum
    except ImportError:
        print("Optimum is not installed. Please install requirements first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "ov_model"
    
    print(f"\nExporting {model_id} to OpenVINO INT8 format in '{output_dir}'...")
    
    # Run the optimum-cli export command for INT8 quantization
    command = [
        "optimum-cli", "export", "openvino",
        "--model", model_id,
        "--weight-format", "int8",
        output_dir
    ]
    
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"\nSuccess! OpenVINO model exported to '{output_dir}'.")
        print("You can now run the application using: python app.py")
    except subprocess.CalledProcessError as e:
        print(f"\nError exporting model: {e}")
        print("Please try running the command manually:")
        print(f"optimum-cli export openvino --model {model_id} --weight-format int8 {output_dir}")

if __name__ == "__main__":
    main()
