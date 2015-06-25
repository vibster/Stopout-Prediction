#utilities
import feature_dict
import getpass
import datetime

#predict
import predictor as predictor

#record
import record_experiments as record

def main(dbName=None, userName=None, passwd=None, dbHost=None,
        dbPort=None,training_course=None, testing_course=None,
        earliest_date=None,latest_date_object=None,features_to_skip=None,
        pred_week=None,feat_week=None, num_weeks=None,epsilon=None,lamb=None):
    if not dbHost:
        dbHost = 'alfa6.csail.mit.edu'
    if not dbPort:
        dbPort = 3306
    if not userName:
        userName='sebboyer'
    if not passwd:
        passwd = getpass.getpass()
    if not dbName:
        dbName = '3091x_2013_spring'
    if not latest_date_object:
        latest_date_object = datetime.datetime.now()
        latest_date = latest_date_object.isoformat()
    if not earliest_date:
        #in seconds:
        dateSlack = 14400 # 2 hours between currentDate and when feature extraction started
        earliest_date = (latest_date_object - datetime.timedelta(seconds=dateSlack)).isoformat()
    if not training_course:
        training_course=dbName
    if not testing_course:
        testing_course=dbName

    if not features_to_skip:
        features_to_skip = [3,4,5,14, 103,104,105, 201, 301, 302]
    features = feature_dict.featuresFromFeaturesToSkip(features_to_skip)

    if not num_weeks:
        #try with 15,16,17
        num_weeks = 15
    weeks = range(num_weeks)

    if not pred_week:
        pred_week=5
    if not feat_week:
        feat_week=3

    if not epsilon:
        epsilon=1
    if not lamb:
        lamb = 1

    #TODO: make sure defaults are correct in feature_dict
    auc_test, auc_train, weights=predictor.run_dropout_prediction(userName,
                                                        passwd, dbHost, dbPort,
                                                        training_course,
                                                        testing_course,
                                                        earliest_date,
                                                        latest_date_object,
                                                        features,
                                                        weeks,
                                                        pred_week,
                                                        feat_week,
                                                        epsilon,
                                                        lamb=lamb)
    print "done"

    #save experiment and model
    #print "Saving run"
    #exp_id = record.record_experiment(dbName, userName, passwd, dbHost, dbPort, pred_week, feat_week,
            #auc_train, testing_course, auc_test, lamb, epsilon, latest_date)

    #record.record_model(dbName, userName, passwd, dbHost, dbPort, features, weights, exp_id)
    #print "done"





if __name__ == "__main__":
    main(dbName='201x_2013_spring',
        features_to_skip = [3,4,5,14,17,103,104,105, 201,204,205,206,207,301,302],
            earliest_date='2015-06-25T10:35:00',
            latest_date_object=datetime.datetime(2015,6,25,17,36,00),
            pred_week = 5,
            feat_week = 3)
