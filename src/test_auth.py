from auth import authenticate_user


if __name__ == "__main__":
    if authenticate_user():
        print("Authentication test passed.")
    else:
        print("Authentication test failed.")