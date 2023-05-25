Steps to composing of Jupyter and PostgreSQL

1) Create a `docker-compose.yml` file
2) Add two services named `jupyter and db`
3) add respective service parameters for jupyter and db
4) for eg: create networks jupyter and db and bind with volume
5) run the `postgres` query to get the column names of csv a.k.a features
6) The notebook names `postgres_jupyter.ipynb` has the illustration of the above points

Steps To Configure ClearML

1) Get the access token from ClearML app through Browser and put them as a cell in Jupyter notebook `clearml.browser_login()`
2) Create a task to perform model `training` on the data from lab02 `task = Task.init()`
3) Choose parameters for HPO `task.connect(params)`
4) Close the task for `training` with `task.close()`

Steps to Perform HPO

1) Everything is a task in ClearML; so create a task for HPO `optimizer=HyperParameterOptimizer()`
2) Define the parameters for this method
3) Make sure you set these commands
```
xgb_optimizer.set_report_period(0.1)
xgb_optimizer.start()  # Start the Optimizer to search best params
xgb_optimizer.wait()   # Give some rest after few minutes of working  
xgb_optimizer.stop()   # Stop the Optimizer when search is done
```
4) Use the below code to get the top 3 options which produced the lowest logloss on train
```
take_items = lambda d, n: [print(f"{k}: {d[k]}") for k in list(d.keys())[:n]]
k = 3
top_exp = optimizer.get_top_experiments(top_k=k)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print("------------------------------------------------")
    print(f"Rank = {n}: task_id = {t.id}")
    print("------------------------------------------------")
    take_items(t.get_parameters(), len(t.get_parameters()))
```