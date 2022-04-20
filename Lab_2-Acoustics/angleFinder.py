from subprocess import call
from time import sleep

N_runs = int(input("How many times do you want to run the test?"))


for i in range(N_runs):
    call(r'angleFinder.bat')
    print(f"Test {i} finished, waiting one minute.")
    sleep(3)