import os

def clear_terminal():
	os.system('cls' if os.name == 'nt' else 'clear')

def add_new_face():
	print("ok")

def main_menu():
	selection = 0
	while True:
		print("|----------------------|")
		print("| 1. Show faces        |")
		print("| 2. Add new face      |")
		print("| 3. Update face data  |")
		print("| 4. Remove face       |")
		print("|----------------------|")
		print("| 5. Start Recognizer  |")
		print("|----------------------|")
		print("| 0. Exit              |")
		print("|----------------------|")
		selection = int(input("select an option: "))
		clear_terminal()

		match selection:
			case 1:
				continue
			case 2:
				add_new_face()
			case 3:
				continue
			case 4:
				continue
			case 5:
				continue
			case 0:
				exit(0)
			case _:
				print("Invalid option selected.")

if __name__ == "__main__":
	main_menu()
