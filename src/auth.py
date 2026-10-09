import getpass
import os

from dotenv import load_dotenv

load_dotenv()


def authenticate_user():
    expected_user_id = os.getenv("APP_USER_ID")
    expected_password = os.getenv("APP_PASSWORD")

    try:
        max_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", "3"))
    except ValueError:
        max_attempts = 3

    if not expected_user_id or not expected_password:
        print("=" * 72)
        print("AUTHENTICATION CONFIGURATION ERROR")
        print("=" * 72)
        print("User ID or password configuration is missing from .env.")
        print("Program terminated.")
        return False

    print()
    print("=" * 72)
    print("SUSTAINABLE MANUFACTURING ANALYTICS")
    print("USER AUTHENTICATION")
    print("=" * 72)

    for attempt in range(1, max_attempts + 1):
        print()

        user_id = input("User ID: ").strip()
        password = getpass.getpass("Password: ")

        if user_id == expected_user_id and password == expected_password:
            print()
            print("=" * 72)
            print("AUTHENTICATION SUCCESSFUL")
            print("=" * 72)
            print("Access granted.")
            print()
            return True

        remaining = max_attempts - attempt

        print()
        print("Invalid User ID or Password.")

        if remaining > 0:
            print(f"Attempts remaining: {remaining}")
        else:
            print("=" * 72)
            print("MAXIMUM LOGIN ATTEMPTS EXCEEDED")
            print("ACCESS DENIED")
            print("PROGRAM TERMINATED")
            print("=" * 72)

    return False