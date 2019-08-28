import logging


def setup_logging(log_level):
    ext_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=log_level)
    if log_level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


def main():
    from parkloader import ParkLoader
    from sys import argv

    try:
        file_name = argv[1]
    except IndexError:
        logging.error("Please provide parkinsons data directory path")
        return

    prl = ParkLoader(file_name)
    logging.info(prl.names)
    for name in prl.names:
        data = prl.load(name)
        print(name, "Labels: ", sorted(set(data.data.label)))
        for train, test, train_lbls, test_lbls, pid in data.leave_one_out(True):
            assert(len(train) == len(train_lbls))
            assert(len(test) == len(test_lbls))
            print(pid)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    main()
