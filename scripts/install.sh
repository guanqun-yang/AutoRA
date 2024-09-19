ask_user_for_setup() {
    echo "Do you want to proceed with GPU setup or CPU setup?"
    echo "Enter 1 for GPU setup, 2 for CPU setup:"

    # read user choice
    read -r choice

    # handle user input
    case $choice in
        1)
            echo "You selected GPU setup."
            gpu_setup
            ;;
        2)
            echo "You selected CPU setup."
            cpu_setup
            ;;
        *)
            echo "Invalid choice. Please enter 1 for GPU or 2 for CPU."
            ask_user_for_setup
            ;;
    esac
}

gpu_setup() {
    echo "Performing GPU setup..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install faiss-gpu
    echo "GPU setup complete!"
}

# Dummy function for CPU setup
cpu_setup() {
    echo "Performing CPU setup..."
    pip install torch torchvision torchaudio
    pip install faiss-cpu
    echo "CPU setup complete!"
}

# reinstall ipython
echo "Reinstall ipython"
conda install ipython

# ask user for installation choice
ask_user_for_setup