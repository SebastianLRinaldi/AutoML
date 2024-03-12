# @title Automatic_ML_Model Class { form-width: "1000px", display-mode: "both" }
# model name
# model to run
# train, test split
# columns
# onehot encoding
# classifaction or continuous
# Clean data (remove na)
# remove columns I choose
# evaulate model (train,dev,test)



# run feature ayanliss then remove those bad columns some how make this automatic??
"""
LinearRegression
LogisticRegression
MLPClassifier # Classifaction Model
MLPRegressor # Continuous Model
RandomForestRegressor
RandomForestClassifier
"""



class automatic_ML_model:
    def __init__(this_model, model_name, model, data_file, target_y_column_name, test_size=0.2, ordinal_map=None, datetime_columns=None, columns_to_remove=None, onehot_encode=True, classification=True):
        # Model-related variables
        this_model.model_name = model_name
        this_model.model = model

        # Data-related variables
        this_model.data_file = data_file
        this_model.df = None
        this_model.test_size = test_size
        this_model.columns_to_remove = columns_to_remove
        this_model.ordinal_map = ordinal_map
        this_model.datetime_columns = datetime_columns
        this_model.onehot_encode = onehot_encode
        this_model.classification = classification

        # Feature engineering and preprocessing-related variables
        this_model.all_columns = None # All columns with input and output features
        this_model.target_y_column_name = target_y_column_name # target variable or the dependent variable, learn and make predictions about

        # Fitted/Trained/Test/prediction model variables
        this_model.X = None # input features or independent variables, uses to make predictions
        this_model.y = None # target variable or the dependent variable, learn and make predictions about

        this_model.X_train = None
        this_model.X_test = None
        this_model.y_train = None
        this_model.y_test = None

        this_model.y_pred_train = None
        this_model.y_pred_test = None

        # Model Validation Metrics
        this_model.cross_validation_score = None
        this_model.confusion_matrix = None

        # Categorical Model Metrics
        this_model.precision_train = None
        this_model.recall_train = None
        this_model.f1_train = None
        this_model.accuracy_train = None

        this_model.precision_test = None
        this_model.recall_test = None
        this_model.f1_test = None
        this_model.accuracy_test = None

        # Continuous Model Metrics
        this_model.mse_train = None
        this_model.mae_train = None
        this_model.mape_train = None
        this_model.r2_train = None

        this_model.mse_test = None
        this_model.mae_test = None
        this_model.mape_test = None
        this_model.r2_test = None




    ####################################################################
    # PART 1 - Load data, Load all Columns, Discover info on columns
    ####################################################################


    def load_datafile_into_dataframe(this_model, file_delimiter=None):
        """
        STEP 1 LOAD DATA as a Data Frame
        """
        this_model.df = pd.read_csv(this_model.data_file, delimiter=file_delimiter)


    def load_all_columns(this_model):
        """
        STEP 2 LOAD ALL Columns into Model Class
        (After discovering)
        """
        # instead of manually doing this I could
        # figure out how to parse the columns from
        # the data frame into a list correctly
        this_model.all_columns = this_model.df.columns


    def discover_all_dataframe_columns(this_model):
        print(this_model.df.columns)

    def discover_all_dataframe_columns_with_info(this_model):
        print("*Too many column will not return column table*")
        print(this_model.df.info())

    def discover_print_all_dataframe_columns(this_model):
        for column in this_model.df.columns:
            print(f"COLUMN NAME: {column}")

    def discover_described_details_about_a_column(this_model, column_name):
        print("\n--------------------------------------------")
        print(f"COLUMN NAME: {column_name}")
        print(this_model.df[column_name].describe())

    def discover_described_details_about_all_columns(this_model):
        # columns = ['buying', 'maintenance', 'doors', 'people', 'boot', 'safety', 'quality']
        for column in this_model.all_columns:
            print("\n--------------------------------------------")
            print(f"COLUMN NAME: {column}")
            print(this_model.df[column].describe())



    # Should add a for loop function can sorts then print the columns that are
    # Floats or ints
    #vs
    # Ones that are strings or something else
    # so we know which one could need encoding with dummies or ordinal stuff


    ####################################################################
    # PART 1.1 - Discover unique info on columns
    ####################################################################


    def discover_unique_values_in_a_column(this_model, column_name):
        # Get unique values within a speficied column
        """
        # Assuming 'df' is your DataFrame and 'column_name' is the name of the column
        unique_values = df['column_name'].unique()
        """
        unique_values = this_model.df[column_name].unique()
        print(unique_values)


    def discover_number_of_unique_values_in_a_column(this_model, column_name):
        print(this_model.df[column_name].nunique())


    def discover_number_of_unique_values_in_y_target_column(this_model):
        # Get unique number of values within target y column
        this_model.discover_number_of_unique_values_in_a_column(this_model.target_y_column_name)


    def discover_unique_values_in_y_target_column(this_model):
        # Get unique values within target y column
        this_model.discover_unique_values_in_a_column(this_model.target_y_column_name)


    #############################################################################
    # Part 2 - Clean columns, remove columns, map(ordinal), encode(nonnumerics)
    #############################################################################


    def procss_data_and_feature_engineering(this_model, verbose_output=False, rank_encoding=False, random_ordinal_map=False, split_datetime_to_components=False):
        this_model.remove_unneeded_columns()

        if this_model.ordinal_map is not None:
            this_model.apply_ordinal_mappings(verbose_output, rank_encoding, random_ordinal_map)

        if this_model.datetime_columns is not None:
            this_model.handle_date_time_columns()


        this_model.handle_non_numerical_data_all_columns(verbose_output, split_datetime_to_components)


    def remove_unneeded_columns(this_model):
         # Columns not needed
        this_model.df = this_model.df.dropna()
        if this_model.columns_to_remove is not None:
            this_model.df = this_model.df.drop(this_model.columns_to_remove, axis=1)


    def apply_ordinal_mappings(this_model, verbose_output, rank_encoding, random_ordinal_map):
        """THIS ONLY APPLYS TO THE Y TARGET COLUMN"""
        # This needs to happen before dummies are encoded
        # if you don't want artifial rankings to be aplied
        # and you don't have a ordinal map already
        # Label Encoding without Ranking
        if rank_encoding is False and this_model.ordinal_map is None or rank_encoding is False and random_ordinal_map is True:
            if verbose_output:
                print("\nUsing LabelEncoder to make random unranked encodings (changed target y column)\n")

            label_encoder = LabelEncoder()
            this_model.df['encoded_y_target_column'] = label_encoder.fit_transform(this_model.df[this_model.target_y_column_name])
            shuffled_labels = this_model.df['encoded_y_target_column'].values.copy()
            # Shuffle the encoded labels to remove ranking
            np.random.shuffle(shuffled_labels)

            # Assign the shuffled labels back to the DataFrame
            this_model.df['shuffled_encoded_y_column'] = shuffled_labels
            this_model.target_y_column_name = 'shuffled_encoded_y_column'


        elif rank_encoding is True and random_ordinal_map is True:
            if verbose_output:
                print("\nUsing LabelEncoder to make random ranked encoding (changed target y column)\n")
            label_encoder = LabelEncoder()
            this_model.df['encoded_y_target_column'] = label_encoder.fit_transform(this_model.df[this_model.target_y_column_name])
            this_model.target_y_column_name = 'encoded_y_target_column'


        elif rank_encoding is True and this_model.ordinal_map is not None:
            if verbose_output:
                print("\nUsing User defined Ordinal Mapping to make predefined ranked encoding (Target y column is same)\n")
            """
            # Ordinal Encoding
            - np.select and np.where are efficient for multiple conditions
            - df.apply() with a custom function offers flexibility for complex logic
            - dictionary mapping method is useful for simple mappings
            - loc is straightforward for direct assignments based on a condition

            # Simple mapping example for quality column
            quality_dict = {
                'vgood': 4,
                'good': 3,
                'acc': 2,
                'unacc': 1,
            }
            """
            # df[column_name_to_map] = df[column_name_to_map].map(mapping_to_be_applied).fillna(0)

            this_model.df[this_model.target_y_column_name] = this_model.df[this_model.target_y_column_name].map(this_model.ordinal_map).fillna(0)

    def handle_date_time_columns(this_model):
        columns = this_model.datetime_columns
        for column in columns:
            this_model.df[column] = pd.to_datetime(this_model.df[column], errors='coerce')
        # print("FOUND TIME COLUMNS")
        this_model.discover_all_dataframe_columns_with_info()


    # def handle_non_numerical_data_all_columns(this_model, show_columns=False):
    #     # Categorical features - dummies encoding
    #     columns = this_model.df.columns.values
    #     for column in columns:
    #         if show_columns:
    #             print("\n-------------------handling non numerics-------------------------")
    #             print(f"COLUMN NAME: {column}")
    #             print(this_model.df[column])
    #         if this_model.df[column].dtype != np.int64 and this_model.df[column].dtype != np.float64:
    #             this_model.df = pd.get_dummies(this_model.df, columns=[column], prefix=column)

    def handle_non_numerical_data_all_columns(this_model, verbose_output, split_datetime_to_components):
        # Categorical features - dummies encoding
        columns = this_model.df.columns.values
        for column in columns:
            if verbose_output:
                # NEED TO ADD a thing that has type of the column before and after handling
                print("\n-------------------handling non numerics-------------------------")
                print(f"COLUMN NAME: {column}")
                # print(this_model.df[column])
            if this_model.df[column].dtype != np.int64 and this_model.df[column].dtype != np.float64:
                # Check if the column is of datetime type

                if pd.api.types.is_datetime64_any_dtype(this_model.df[column]):
                    if verbose_output:
                        print("If you get an error about 'DType could not be promoted'")
                        print("Need to set split_datetime_to_components=True")

                    if split_datetime_to_components:
                        # Extract components from datetime
                        this_model.df[f'{column}_year'] = this_model.df[column].dt.year
                        this_model.df[f'{column}_month'] = this_model.df[column].dt.month
                        this_model.df[f'{column}_day'] = this_model.df[column].dt.day
                        this_model.df[f'{column}_hour'] = this_model.df[column].dt.hour
                        # Drop the original datetime column
                        this_model.df.drop(column, axis=1, inplace=True)
                else:
                    this_model.df = pd.get_dummies(this_model.df, columns=[column], prefix=column)
                    # # Apply pd.get_dummies with drop_first to reduce dimensionality
                    # dummies = pd.get_dummies(this_model.df[column], prefix=column, drop_first=True)
                    # # Concatenate the original DataFrame with the new dummy variables
                    # this_model.df = pd.concat([this_model.df, dummies], axis=1)
                    # # Drop the original non-numerical column
                    # this_model.df.drop(column, axis=1, inplace=True)
        if verbose_output:
            print("\n--------------New Columns After Handling Non Numerics-------------------------")
            this_model.discover_all_dataframe_columns_with_info()



    #############################################################################
    # Part 3 - Train and Fit the Model
    #############################################################################


    def train_the_model(this_model, verbose_output=False, stratify_on=False):
        """Need to add a test/train/dev (validation set) here"""
        # Step 3: Model Training

        # Model training
        if verbose_output:
            print("Now Training model...")
            print("Note: Anything related to converagnce or interation,")
            print("\tadjust hyperpameters of model=model()\n")
        this_model.X = this_model.df.drop([this_model.target_y_column_name], axis=1) # Just the x cloumns
        this_model.y = this_model.df[this_model.target_y_column_name] # Just the Target y column
        this_model.fitted_model = this_model.model.fit(X=this_model.X, y=this_model.y)
        if stratify_on == True:
            X_train, X_test, y_train, y_test = train_test_split(this_model.X, this_model.y, test_size=this_model.test_size, random_state=42, stratify=this_model.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(this_model.X, this_model.y, test_size=this_model.test_size, random_state=42)

        this_model.X_train = X_train
        this_model.X_test = X_test
        this_model.y_train = y_train
        this_model.y_test = y_test
        if verbose_output:
            print("\nDone Training Model!\n")


    #############################################################################
    # Part 4 - Cross Validation
    #############################################################################
    def perform_cross_valaidation(this_model, verbose_output=False, cv_number_of_folds=5):
        # Step 4: Cross-Validation
        if verbose_output:
            print(f"\nStart Cross Validation with {cv_number_of_folds}...")
            print("Note: Anything related to converagnce or interation,")
            print("\tadjust the number of cross validation folds \n")
        this_model.cross_validation_score = cross_val_score(this_model.model,  this_model.X,  this_model.y, cv=cv_number_of_folds)
        if verbose_output:
            print("End Cross Validation...\n")


    #############################################################################
    # Part 5 - Make predictions with model
    #############################################################################
    def perform_predictions_on_model(this_model):
        # Step 5: Predict with the model
        # For the evaluation matrics to perform the evals on train and test
        # Just have a logic that cahnged the values for all the functions
        # from useing training and train predicting to use testing and training predicting
        this_model.y_pred_test =  this_model.model.predict(this_model.X_test)
        this_model.y_pred_train =  this_model.model.predict(this_model.X_train)


    #############################################################################
    # Part 6 - Performance Evaluation Metrics
    #############################################################################
    def perform_categorical_evluation_scores(this_model, multi_class=True):
        """
        Evaluate the model's performance for classification tasks.
        """
        # Am I working with a Multi-class or binary class?
        """
        average='binary' - for two class classifaction
        average='macro' - for more than two class classifaction
        """
        # Evaluate the model's performance
        # This means we are working with a classifaction model
        if multi_class == True:
            average_class = 'macro'
        elif multi_class == False:
            average_class='binary'

        this_model.precision_test = precision_score(this_model.y_test, this_model.y_pred_test, average=average_class)

        this_model.recall_test = recall_score(this_model.y_test, this_model.y_pred_test, average=average_class)

        this_model.f1_test = f1_score(this_model.y_test, this_model.y_pred_test, average=average_class)

        this_model.accuracy_test = accuracy_score(this_model.y_test, this_model.y_pred_test)


        # Set training
        this_model.precision_train = precision_score(this_model.y_train, this_model.y_pred_train, average=average_class)

        this_model.recall_train = recall_score(this_model.y_train, this_model.y_pred_train, average=average_class)

        this_model.f1_train = f1_score(this_model.y_train, this_model.y_pred_train, average=average_class)

        this_model.accuracy_train = accuracy_score(this_model.y_train, this_model.y_pred_train)



    def perform_continuous_evaluation_scores(this_model):
        """
        Evaluate the model's performance for regression tasks.
        """
        # Compute evaluation metrics
        # this_model.accuracy = accuracy_score(this_model.y_test, this_model.y_pred_test) # This might break (if it doesn't add in everything but precsion)

        this_model.mse_test = mean_squared_error(this_model.y_test, this_model.y_pred_test)

        # this_model.rmse = np.sqrt(mean_squared_error(this_model.y_true, this_model.y_pred))

        this_model.mae_test = mean_absolute_error(this_model.y_train, this_model.y_pred_train)

        this_model.mape_test = mean_absolute_percentage_error(this_model.y_test, this_model.y_pred_test)

        this_model.r2_test = r2_score(this_model.y_test, this_model.y_pred_test)
        # model.score(this_model.y_test, this_model.y_pred_test)

        this_model.mse_train = mean_squared_error(this_model.y_train, this_model.y_pred_train)

        this_model.mae_train = mean_absolute_error(this_model.y_train, this_model.y_pred_train)

        this_model.mape_train = mean_absolute_percentage_error(this_model.y_train, this_model.y_pred_train)

        this_model.r2_train = r2_score(this_model.y_train, this_model.y_pred_train)



    def make_confusion_matrix(this_model):
        # Step 7: Confusion Matrix
        this_model.confusion_matrix = confusion_matrix(this_model.y_test, this_model.y_pred_test)


    #############################################################################
    # Part 7.1 - Show Performance Evaluation Metrics (Confusion Matrix)
    #############################################################################
    def discover_prediction_columns(this_model):
        this_model.discover_unique_values_in_y_target_column()

    def show_cm(this_model, prediction_columns):
        """NEED TO PASS IN THE COLUMNS FOR CM"""
        # Create a DataFrame for better visualization of what is being manipulated
        """For this I feel like we just pass in the ordinal mapping as a list? but of just the keys???"""

        print('Confusion Matrix:\n', this_model.confusion_matrix)
        df_cm = pd.DataFrame(this_model.confusion_matrix, index=prediction_columns, columns=prediction_columns)

        # Plot the confusion matrix with calculated values
        plt.figure(figsize=(10,7))
        sns.heatmap(df_cm, annot=True, fmt="d", square=True, cmap='Blues_r')

        # Add labels to the heatmap
        for x in range(df_cm.shape[0]):
            for y in range(df_cm.shape[1]):
                plt.text(x, y, df_cm.iloc[x, y], ha='left', va='top', color='red')

        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('Confusion Matrix')
        plt.show()

    #############################################################################
    # Part 7.2 - Show Performance Evaluation Metrics (categorical or continuous)
    #############################################################################
    def display_categorical_metrics(this_model):
        """
        Validation Metrics
        """
        if this_model.cross_validation_score is not None:
            print("\n-------------Cross Validation Metrics--------------")
            print(f"\n**************{this_model.model_name}*****************")
            print(f"Cross-validated scores: {this_model.cross_validation_score}")
            print(f"Mean cross-validated score: {this_model.cross_validation_score.mean():.3f}")
        # Could set up a dictionary that basically would hold the metrics
        # then her ewe just have a for loop that says
        # for each metric val in this_model.metrics_dict
        #   print the metric if not none

        """
        Categorical Metrics
        """
        print("\n-------------Categorical Metrics--------------")
        print(f"\n**************{this_model.model_name}*****************")
        # Categorical Model Metrics for Training
        print(f"Model Accuracy (Training): {this_model.accuracy_train:.3f}")
        print(f"Precision (Training): {this_model.precision_train:.3f}")
        print(f"Recall (Training): {this_model.recall_train:.3f}")
        print(f"F1 Score (Training): {this_model.f1_train:.3f}\n")

        # Categorical Model Metrics for Testing
        print(f"Model Accuracy (Testing): {this_model.accuracy_test:.3f}")
        print(f"Precision (Testing): {this_model.precision_test:.3f}")
        print(f"Recall (Testing): {this_model.recall_test:.3f}")
        print(f"F1 Score (Testing): {this_model.f1_test:.3f}\n")


    def display_continuous_metrics(this_model):
        """
        Validation Metrics
        """
        if this_model.cross_validation_score is not None:
            print("\n-------------Cross Validation Metrics--------------")
            print(f"\n**************{this_model.model_name}*****************")
            print(f"Cross-validated scores: {this_model.cross_validation_score}")
            print(f"Mean cross-validated score: {this_model.cross_validation_score.mean():.3f}")

        """
        Continuous Metrics
        """
        # Continuous Model Metrics for Training
        print("\n-------------Continuous Metrics--------------")
        print(f"\n**************{this_model.model_name}*****************")
        print(f"Mean Squared Error (MSE) (Training) - Train: {this_model.mse_train:.3f}")
        print(f"Mean Absolute Error (MAE) (Training) - Train: {this_model.mae_train:.3f}")
        print(f"Mean Absolute Percentage Error (MAPE) (Training) - Train: {this_model.mape_train:.3f}")
        print(f"R-squared (R2) (Training)- Train: {this_model.r2_train:.3f}")

        # Continuous Model Metrics for Testing
        print(f"Mean Squared Error (MSE) (Testing) - Test: {this_model.mse_test:.3f}")
        print(f"Mean Absolute Error (MAE) (Testing) - Test: {this_model.mae_test:.3f}")
        print(f"Mean Absolute Percentage Error (MAPE) (Testing) - Test: {this_model.mape_test:.3f}")
        print(f"R-squared (R2) (Testing) - Test: {this_model.r2_test:.3f}\n")



    #############################################################################
    # Part 8 - Compare Performance Evaluation Metrics Against Other Metrics
    #############################################################################
    def run_adavanced_metrics(this_model):

        # Predict probabilities
        y_probs = this_model.model.predict_proba(this_model.X_test)[:, 1]

        # Calculate ROC AUC
        print("ROC AUC is a performance measurement for classification problems at various thresholds settings.\n"
                "It tells how much a model is capable of distinguishing between classes.\n"
                "The higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s.\n")
        auc = roc_auc_score(this_model.y_test, y_probs)
        print('AUC: %.3f\n' % auc)

        # Calculate Brier Score
        print("The Brier score is a measure of the accuracy of probabilistic predictions.\n"
                "It calculates the mean squared error between the predicted probabilities and the actual outcomes.\n"
                "A lower Brier score indicates better predictive accuracy.\n")
        brier_score = brier_score_loss(this_model.y_test, y_probs)
        print('Brier Score: %.3f\n' % brier_score)


    """
    evaluate your model across different metrics, you can use the cross_validate function from
    sklearn.model_selection with a custom scoring function if necessary.
    For example, to evaluate a model using multiple metrics:
    """
    # from sklearn.model_selection import cross_validate
    # from sklearn.metrics import make_scorer, mean_squared_error, r2_score

    # # Define a custom scoring function if needed
    # mse_scorer = make_scorer(mean_squared_error)
    # r2_scorer = make_scorer(r2_score)

    # # Assuming clf is your classifier and X, y are your features and targets
    # scoring = {'mse': mse_scorer, 'r2': r2_scorer}
    # cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

    # # Accessing the results
    # print(cv_results['test_mse'])
    # print(cv_results['test_r2'])


    #############################################################################
    # Part 9 - Compare Performance Evaluation Metrics of Two Different Models
    #############################################################################


    #### Compare metrics of Model 1 to Model 2 (implment as well a graph that shows)
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
def compare_stats_catigorical_models(first_model, second_model):
    # IMplement so that >, < or == get printed per each stat (extra)

    print("\n===============Comparing Model Stats================")
    print(f"      {first_model.model_name} -VS- {second_model.model_name}")
    print(f"precision_model_one (Test) {first_model.precision_test:.3f} | precision_model_two (Test) {second_model.precision_test:.3f}")
    print(f"precision_model_one (Train) {first_model.precision_train:.3f} | precision_model_two (Train) {second_model.precision_train:.3f}\n")
    print(f"   recall_model_one (Test) {first_model.recall_test:.3f} | recall_model_two (Test) {second_model.recall_test:.3f}")
    print(f"   recall_model_one (Train) {first_model.recall_train:.3f} | recall_model_two (Train) {second_model.recall_train:.3f}\n")
    print(f"       f1_model_one (Test) {first_model.f1_test:.3f} | f1_model_two (Test) {second_model.f1_test:.3f}")
    print(f"       f1_model_one (Train) {first_model.f1_train:.3f} | f1_model_two (Train) {second_model.f1_train:.3f}\n")
    print(f" accuracy_model_one (Test) {first_model.accuracy_test:.3f} | accuracy_model_two (Test) {second_model.accuracy_test:.3f}")
    print(f" accuracy_model_one (Train) {first_model.accuracy_train:.3f} | accuracy_model_two (Train) {second_model.accuracy_train:.3f}")


    if first_model.cross_validation_score is not None:
            print(f"\n-------------Cross Validation Metrics {first_model.model_name} --------------")
            print(f"Cross-validated scores: {first_model.cross_validation_score}")
            print(f"Mean cross-validated score: {first_model.cross_validation_score.mean():.3f}\n")

    if second_model.cross_validation_score is not None:
            print(f"\n-------------Cross Validation Metrics {second_model.model_name} --------------")
            print(f"Cross-validated scores: {second_model.cross_validation_score}")
            print(f"Mean cross-validated score: {second_model.cross_validation_score.mean():.3f}\n")


def compare_stats_continous_models(first_model, second_model):

    print("\n===============Comparing Model Stats================")
    print(f"      {first_model.model_name} -VS- {second_model.model_name}")
    print(f"Mean Squared Error (MSE) (Test) {first_model.mse_test:.3f} | Mean Squared Error (MSE) (Test) {second_model.mse_test:.3f}")
    print(f"Mean Squared Error (MSE) (Train) {first_model.mse_train:.3f} | Mean Squared Error (MSE) (Train) {second_model.mse_train:.3f}\n")
    print(f"Mean Absolute Error (MAE) (Test) {first_model.mae_test:.3f} | Mean Absolute Error (MAE) (Test) {second_model.mae_test:.3f}")
    print(f"Mean Absolute Error (MAE) (Train) {first_model.mae_train:.3f} | Mean Absolute Error (MAE) (Train) {second_model.mae_train:.3f}\n")
    print(f"Mean Absolute Percentage Error (MAPE) (Test) {first_model.mape_test:.3f} | Mean Absolute Percentage Error (MAPE) (Test) {second_model.mape_test:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE) (Train) {first_model.mape_train:.3f} | Mean Absolute Percentage Error (MAPE) (Train) {second_model.mape_train:.3f}\n")
    print(f"R-squared (R2) (Test) {first_model.r2_test:.3f} | R-squared (R2) (Test {second_model.r2_test:.3f}")
    print(f"R-squared (R2) (Train) {first_model.r2_train:.3f} | R-squared (R2) (Train) {second_model.r2_train:.3f}\n")


    if first_model.cross_validation_score is not None:
            print(f"\n-------------Cross Validation Metrics {first_model.model_name} --------------")
            print(f"Cross-validated scores: {first_model.cross_validation_score}")
            print(f"Mean cross-validated score: {first_model.cross_validation_score.mean():.3f}\n")

    if second_model.cross_validation_score is not None:
            print(f"\n-------------Cross Validation Metrics {second_model.model_name} --------------")
            print(f"Cross-validated scores: {second_model.cross_validation_score}")
            print(f"Mean cross-validated score: {second_model.cross_validation_score.mean():.3f}\n")
    #############################################################################
    # Part 10 - Feature importance, then retaining the model
    #############################################################################


def feature_importance_plot_and_retrain(this_model):
    print("Running feature importance...could be a while")
    result = permutation_importance(this_model.model, this_model.X_test, this_model.y_test, n_repeats=60, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    print("Finished feature importance")
    print("Showing feature Importance")
    fig, ax = plt.subplots(figsize=(15,15))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(this_model.X.columns)[sorted_idx])
    plt.title("Permutation Importances")

    plt.show()

    # Get the indices of the top 10 features
    # number_of_top_features_to_get = input("Number of top features to get: ")
    # top_features = result.importances_mean.argsort()[-57:]

    # # Create a new DataFrame with only the top 20 features
    # this_model.X = this_model.X[np.array(this_model.X.columns)[top_features]]
    """Next step would be the retrain the model"""


def retrain_the_model(this_model, stratify_on=False):
        """Need to add a test/train/dev (validation set) here"""

        # Model training
        print("Now reTraining model...")
        print("Note: Anything related to converagnce or interation,")
        print("\tadjust those where you define the model\n")

        this_model.fitted_model = this_model.model.fit(this_model.X, this_model.y)
        if stratify_on == True:
            X_train, X_test, y_train, y_test = train_test_split(this_model.X, this_model.y, test_size=this_model.test_size, random_state=42, stratify=this_model.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(this_model.X, this_model.y, test_size=this_model.test_size, random_state=42)

        this_model.X_train = X_train
        this_model.X_test = X_test
        this_model.y_train = y_train
        this_model.y_test = y_test

        print("\nDone reTraining Model!\n")


