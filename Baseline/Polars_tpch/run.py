from src.utils.timerutil import TPCHTimer

if __name__ == "__main__":
    with TPCHTimer("test", True):
        print("hello")
