import importlib, sys, json, io, time, traceback, itertools, os
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime, timedelta
from collections import namedtuple
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# hitting lower accuracy corresponds to grade of 0%
# hitting upper accuracy corresponds to grade of 100%
lower = 50
upper = 75
max_sec = 60
module_name = "main"
test_set = "test1"

if len(sys.argv) > 3:
    print("Usage: python3 test.py [mod_name] [test_set]")
    sys.exit(1)
if len(sys.argv) > 1:
    module_name = sys.argv[1]
if len(sys.argv) > 2:
    test_set = sys.argv[2]

student_module = importlib.import_module(module_name)

def main():
    print(f"We'll use this to grade you:\n    python3 tester.py main test2\n")
    print("test2_users.csv and test2_log.csv are secret, so you can use this to estimate your accuracy:\n    python3 tester.py main test1\n")

    print("Grading:")
    print(f"    Max Seconds: {max_sec}")
    print(f"    Accuracy <{lower}: grade=0%")
    print(f"    Accuracy >{upper}: grade=100%")
    print()

    t0 = time.time()
    print("Fitting+Predicting...")

    # step 1: fit
    model = student_module.FraudDetector()
    train_customers = pd.read_csv(os.path.join("data", "train_customers.csv"))
    train_transactions = pd.read_csv(os.path.join("data", "train_transactions.csv"))
    train_is_fraud = pd.read_csv(os.path.join("data", "train_is_fraud.csv"))
    model.fit(train_customers, train_transactions, train_is_fraud)

    # step 2: predict
    test_customers = pd.read_csv(os.path.join("data", "{}_customers.csv".format(test_set)))
    test_transactions = pd.read_csv(os.path.join("data", "{}_transactions.csv".format(test_set)))
    test_transactions = test_transactions.sort_values(by=['trans_num'], ignore_index=True)
    y_pred = model.predict(test_customers, test_transactions)

    # step 3: grading based on accuracy
    y = pd.read_csv(os.path.join("data", "{}_is_fraud.csv".format(test_set)))
    y = y.sort_values(by=['trans_num'], ignore_index=True)
    y = np.array(y["is_fraud"])
    accuracy = balanced_accuracy_score(y, y_pred)
    grade = round((np.clip(accuracy, lower, upper) - lower) / (upper - lower) * 100, 1)

    t1 = time.time()
    sec = t1-t0
    assert sec < max_sec
    warn_sec = 0.75 * max_sec
    if sec > warn_sec:
        print("="*40)
        print("WARNING!  Tests took", sec, "seconds")
        print("Maximum is ", max_sec, "seconds")
        print(f"We recommend keeping runtime under {warn_sec} seconds to be safe.")
        print("Variability may cause it to run slower for us than you.")
        print("="*40)

    # output results
    results = {"score":grade,
               "accuracy": accuracy,
               "date":datetime.now().strftime("%m/%d/%Y"),
               "latency": sec}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Result:\n" + json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
