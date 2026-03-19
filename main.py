print("Colon Disease Detection System")

print("1. Predict using Image")
print("2. Predict using Text")

choice = input("Enter choice: ")

if choice == "1":
    import src.predict_image

elif choice == "2":
    import src.predict_text

else:
    print("Invalid choice")