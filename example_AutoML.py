RandomForestClassifier_AutoModel_Default = automatic_ML_model(
    model_name = 'RandomForestClassifier',
    model = RandomForestClassifier(n_estimators=100, random_state=0),
    data_file = 'Data.csv',
    target_y_column_name = 'Returned',
    ordinal_map = None,
    datetime_columns = ['OrderDate', 'CustomerBirthDate'],
    test_size = 0.8,
    columns_to_remove = ['ID','OrderID','CustomerID','CustomerState'],
    )


def run_RandomForestClassifier_AutoModel_Default(model):
    """STEP 1"""
    model.load_datafile_into_dataframe(
        file_delimiter=None
        )

    model.discover_all_dataframe_columns()

    model.load_all_columns()


    """STEP 2"""

    model.procss_data_and_feature_engineering(
        verbose_output=True,
        rank_encoding=False,
        random_ordinal_map=False,
        split_datetime_to_components=True
    )

    """STEP 3"""
    model.train_the_model(
        verbose_output=True,
        stratify_on=False)

    """STEP 8 - Inspect features"""
    # feature_importance_plot_and_retrain(model)

    """STEP 4"""
    model.perform_cross_valaidation(
        verbose_output=True,
        cv_number_of_folds=5
        )

    """STEP 5"""
    model.perform_predictions_on_model()

    """STEP 6"""
    model.perform_categorical_evluation_scores(
        multi_class=True
        )
    # model.perform_continuous_evaluation_scores()
    model.make_confusion_matrix()

    """STEP 7"""
    model.discover_prediction_columns()
    model.show_cm([0, 1])
    model.display_categorical_metrics()
    # model.display_continuous_metrics()

    model.run_adavanced_metrics()

run_RandomForestClassifier_AutoModel_Default(RandomForestClassifier_AutoModel_Default)
