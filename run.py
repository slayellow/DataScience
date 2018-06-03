from app import app
def main():

    start = app()
    start.load_config()
    start.run_model()



if __name__ == '__main__':
    main()