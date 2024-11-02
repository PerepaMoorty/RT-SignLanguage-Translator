import os, time
start_time = time.time()
os.system('cls' if os.name == 'nt' else 'clear')

from Window_Manager import main
main()

end_time = time.time()
print(f'\nTotal Runtime: {end_time - start_time :.4f} seconds')