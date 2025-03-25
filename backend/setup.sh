#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
print_message() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Function to install a package with retries
install_package() {
    local package=$1
    local max_retries=3
    local retry_count=0
    local success=false
    local extra_args=$2

    while [ "$success" = "false" ] && [ $retry_count -lt $max_retries ]; do
        if [ $retry_count -gt 0 ]; then
            print_warning "Retrying installation of $package (Attempt $((retry_count + 1))/$max_retries)"
            sleep 2
        fi

        if [ -n "$extra_args" ]; then
            if pip install --no-cache-dir $extra_args "$package"; then
                success=true
                print_message "Successfully installed $package"
            else
                retry_count=$((retry_count + 1))
                pip cache purge
            fi
        else
            if pip install --no-cache-dir "$package"; then
                success=true
                print_message "Successfully installed $package"
            else
                retry_count=$((retry_count + 1))
                pip cache purge
            fi
        fi
    done

    if [ "$success" = "false" ]; then
        print_error "Failed to install $package after $max_retries attempts"
        return 1
    fi
    return 0
}

# Function to install ML packages
install_ml_packages() {
    print_message "Installing Machine Learning packages..."
    
    # Try to install torch and torchvision from PyPI first
    if ! install_package "torch==2.0.1" "--index-url https://download.pytorch.org/whl/cu117"; then
        print_warning "Failed to install torch with CUDA support, falling back to CPU version"
        if ! install_package "torch==2.0.1"; then
            print_error "Failed to install torch"
            return 1
        fi
    fi

    if ! install_package "torchvision==0.15.2" "--index-url https://download.pytorch.org/whl/cu117"; then
        print_warning "Failed to install torchvision with CUDA support, falling back to CPU version"
        if ! install_package "torchvision==0.15.2"; then
            print_error "Failed to install torchvision"
            return 1
        fi
    fi

    # Install tensorflow
    if ! install_package "tensorflow==2.13.0"; then
        print_warning "Failed to install tensorflow, trying tensorflow-cpu"
        if ! install_package "tensorflow-cpu==2.13.0"; then
            print_error "Failed to install tensorflow"
            return 1
        fi
    fi

    # Install transformers and sentence-transformers
    if ! install_package "transformers==4.31.0"; then
        print_error "Failed to install transformers"
        return 1
    fi

    if ! install_package "sentence-transformers==2.2.2"; then
        print_error "Failed to install sentence-transformers"
        return 1
    fi

    return 0
}

# Function to install dependencies in stages
install_dependencies() {
    print_message "Installing dependencies in stages..."

    # Core dependencies first
    local core_deps=(
        "typing-extensions==4.5.0"
        "numpy==1.24.3"
        "pandas==2.0.3"
        "scipy==1.10.1"
        "scikit-learn==1.3.0"
        "Pillow==9.5.0"
        "pyarrow==11.0.0"
    )

    print_message "Installing core dependencies..."
    for dep in "${core_deps[@]}"; do
        if ! install_package "$dep"; then
            print_error "Failed to install core dependency: $dep"
            exit 1
        fi
    done

    # Install ML packages separately
    if ! install_ml_packages; then
        print_error "Failed to install ML packages"
        exit 1
    fi

    # Install remaining packages by group
    local groups=(
        "Web Framework"
        "HTTP and API"
        "Data Processing"
        "Development and Testing"
        "Utilities"
        "Database"
        "Security"
    )

    for group in "${groups[@]}"; do
        print_message "Installing $group packages..."
        local packages=$(sed -n "/# $group/,/^$/p" requirements.txt | grep "^[^#]" || true)
        
        if [ -n "$packages" ]; then
            while IFS= read -r package; do
                if [[ ! " ${core_deps[@]} " =~ " ${package} " ]]; then
                    if ! install_package "$package"; then
                        print_warning "Failed to install package from $group: $package"
                    fi
                fi
            done <<< "$packages"
        fi
    done

    # Verify installation
    print_message "Verifying installation..."
    if ! pip check; then
        print_warning "Some package dependencies may have conflicts. Please check the output above."
    else
        print_message "All dependencies installed successfully!"
    fi
}

# Main script execution
main() {
    # Check if Python is installed
    if ! command -v python >/dev/null 2>&1; then
        print_error "Python is not installed. Please install Python first."
        exit 1
    fi

    # Create and activate virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_message "Creating virtual environment..."
        python -m venv venv
    fi

    # Activate virtual environment
    source venv/Scripts/activate || source venv/bin/activate

    # Upgrade pip
    print_message "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install dependencies
    install_dependencies

    print_message "Setup completed successfully!"
}

# Execute main function
main 