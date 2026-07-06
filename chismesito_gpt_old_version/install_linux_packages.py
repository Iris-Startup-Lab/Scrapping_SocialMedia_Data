import subprocess
import sys

def run_bash_command(command_list):
    """
    Runs a bash command.
    :param command_list: A list of strings representing the command and its arguments.
                         e.g., ["apt-get", "install", "-y", "chromium-chromedriver"]
    :return: True if successful, False otherwise.
    """
    try:
        print(f"Executing: {' '.join(command_list)}")
        # `check=True` will raise a CalledProcessError if the command returns a non-zero exit code.
        # `capture_output=True` and `text=True` to get stdout/stderr as strings.
        result = subprocess.run(command_list, check=True, capture_output=True, text=True)
        print("Command executed successfully.")
        if result.stdout:
            print("Stdout:\n", result.stdout)
        if result.stderr:
            print("Stderr:\n", result.stderr) # Some tools print to stderr even on success
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Stdout (error):\n", e.stdout)
        if e.stderr:
            print("Stderr (error):\n", e.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: The command '{command_list[0]}' was not found. Is it in your PATH?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

# --- Example: Installing chromedriver on a Debian/Ubuntu-based system ---
# This is just an example. The actual commands depend on your OS and permissions.

def install_chromedriver_debian_ubuntu():
    """
    Attempts to install chromium-chromedriver on a Debian/Ubuntu system.
    Note: This often requires root privileges. If you are already root (e.g., in some Docker containers),
    you might not need 'sudo'. If 'sudo' is not available and you are not root, this will fail.
    """
    print("Attempting to install chromium-chromedriver (Debian/Ubuntu)...")

    # Check if running as root or if sudo is available and needed.
    # For simplicity, this example tries with sudo first, then without if sudo fails.
    # In a controlled environment (like a Dockerfile or HF Space), you usually know if you're root.

    update_cmd_sudo = ["sudo", "apt-get", "update", "-y"]
    #install_cmd_sudo = ["sudo", "apt-get", "install", "-y", "chromium-chromedriver"]
    install_cmd_sudo = ["sudo", "apt-get", "install", "-y", "chromium-driver"]

    update_cmd_no_sudo = ["apt-get", "update", "-y"]
    install_cmd_sudo = ["apt-get", "install", "-y", "chromium-driver"]
    #install_cmd_no_sudo = ["apt-get", "install", "-y", "chromium-chromedriver"]

    # Try with sudo
    print("\nTrying with sudo...")
    if run_bash_command(update_cmd_sudo) and run_bash_command(install_cmd_sudo):
        print("Installation with sudo successful (or sudo was not needed and commands ran).")
        return True
    
    # If sudo failed (e.g., command not found, or permission denied for sudo itself)
    # and the error wasn't that apt-get failed, try without sudo.
    # This logic can be refined based on specific error messages from run_bash_command.
    print("\nSudo attempt might have failed or was not necessary. Trying without sudo...")
    if run_bash_command(update_cmd_no_sudo) and run_bash_command(install_cmd_no_sudo):
        print("Installation without sudo successful.")
        return True
        
    print("Failed to install chromium-chromedriver using both sudo and non-sudo attempts.")
    return False

# --- How you might use it (conceptual) ---
# if __name__ == "__main__":
#     # This is OS-dependent. For Linux:
#     if sys.platform.startswith('linux'):
#         # You might want to add a check here to see if chromedriver is already installed
#         # and accessible before attempting to install it.
#         # For example, by trying to run `chromedriver --version`
#
#         print("Running on Linux. Checking if chromedriver installation is needed.")
#         if install_chromedriver_debian_ubuntu():
#             print("Chromedriver installation process completed.")
#         else:
#             print("Chromedriver installation process failed.")
#     else:
#         print(f"Automated chromedriver installation not implemented for OS: {sys.platform}")
#         print("Please ensure chromedriver is installed and in your PATH, or use webdriver-manager.")

