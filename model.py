# Python Modules
from datetime import datetime
import typing
# 3rd Party Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from scipy.optimize import minimize
from scipy import stats
import seaborn as sn


def preprocess_inputs(df: pd.DataFrame, zscore: bool = False) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """ Preprocess a dataframe into values useful for modeling purposes.

    Args:
        df: Art market data to preprocess.
        zscore: Whether to standardize to z-score.

    Returns:
        inputs: Preprocessed inputs for modeling.
        outputs: True values.
    """
    # Drop records that are missing estimate_low.
    # We require estimate_low for modeling and there are few instances where it is missing.
    inputs = df.loc[(df['estimate_high'] > 0) & (df['estimate_low'] > 0), :].copy()
    inputs = inputs.drop_duplicates(subset=['auction_sale_id', 'lot_id'], keep='first')

    # Create dummy variables for auction_house variable.
    houses = pd.get_dummies(inputs['auction_house'])

    # Create dummies for auction_locations.
    # Group low occurrence locations into an other group.
    inputs.loc[inputs['auction_location'].isin(('Online', 'Shanghai', 'Hong Kong', 'Amsterdam')), 'auction_location'] = 'Other'
    locations = pd.get_dummies(inputs['auction_location'])

    # Use auction_date to create a linear time column, to capture potential time-dependent trends.
    # inputs['time'] = (pd.to_datetime(inputs['auction_date']) - datetime(2005,1,1)).apply(lambda x: x.days / 365)
    inputs['year'] = pd.to_datetime(inputs['auction_date']).dt.year.astype(str)
    year = pd.get_dummies(inputs['year'])

    # Use lot_id and auction_lot_count to derive a "% of lots in front of this piece" so that its standardized
    # for variables auction lengths.
    # clean_lots = lambda x: int(str(x).replace('A','').replace('B',''))
    # inputs['lot_place_in_auction'] = inputs['lot_id'].apply(clean_lots)
    # inputs['auction_lot_count'] = inputs['auction_lot_count'].apply(clean_lots)
    inputs['lot_place_in_auction'] = inputs['lot_id'].str.replace(r"A|B", "").astype(int)
    inputs['lot_location'] = inputs['lot_place_in_auction'] / inputs[['lot_place_in_auction', 'auction_lot_count']].max(axis=1)

    # TODO Add work medium: Currently too many parameters created here.
    medium = pd.get_dummies(inputs['work_medium'])

    # Use the high / low estimates of price, but convert to mean to minimize multicollinearity.
    # inputs['exp_price'] = np.log((inputs['estimate_high'] + inputs['estimate_low']) / 2)

    # Adjust for currency. We transform price into log price because there is a huge amount of skew in pricing.
    # The skew can make parameter estimation more difficult.
    outputs = pd.DataFrame()
    outputs['hammer_price'] = np.log10(inputs['hammer_price'] * inputs['exchange_rate_to_usd']) # Converts -1's to NaNs.
    outputs['estimate_low_y'] = np.log10(inputs['estimate_low'] * inputs['exchange_rate_to_usd'])

    inputs['estimate_high'] = np.log10(inputs['estimate_high'])
    inputs['estimate_low'] = np.log10(inputs['estimate_low'])

    # Concatenate into new dataframe for inputs and return.
    # include = ['time', 'lot_location', 'exp_price']
    include = ['lot_location', 'estimate_high', 'estimate_low']
    inputs = pd.concat([houses, locations, year, medium, inputs[include],], axis=1)

    # Flag for normalizing inputs on the same scale making for easier model interpretability.
    if zscore:
        for col in include:
            inputs[col] = (inputs[col] - inputs[col].mean()) / inputs[col].std()

    # Column of ones for offset.
    inputs['ones'] = 1 

    return inputs, outputs


def train_model(inputs: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    """ Train model coefficients.

    Args:
        inputs: Art market data to train over.
        outputs: True values.

    Results:
        Trained model.
    """
    # Compute the model. Here we will use a simple linear model, but with a custom loss function.
    # Because we have lots of prices that are missing, we don't want to drop that data - its very important
    # because we know that the price must be LESS than the opening auction bid, but we don't know how much less.
    # I.e. the piece didn't sell because its worth zero - its just worth less than was being asked for it.
    # If hammer price missing, we set loss = 0 if the model predicts a value < estimate_low, and assign
    # regular squared loss in the case that value > estimate_low.  Because of our custom loss function, we can't
    # use OLS to solve for the model coefficients, we must use gradient descent.
    def custom_loss_func(beta, X, outputs):
        """ Custom loss: If hammer price == -1, loss = (yhat - estimate_low)**2 if yhat > estimate_low, else 0. """
        yhat = np.dot(X, beta).transpose()
        loss1 = outputs['hammer_price'].notna()*np.power(yhat - outputs['hammer_price'], 2) 
        loss2 = outputs['hammer_price'].isna()*np.power(np.maximum(yhat - outputs['estimate_low_y'], 0), 2)
        return (loss1.sum() + loss2.sum())

    beta_init = np.array([0]*inputs.shape[1])
    model = minimize(custom_loss_func, beta_init, args=(inputs.values, outputs),
             options={'maxiter': 500}, method='BFGS')

    return model


def train_test_split(inputs: pd.DataFrame, outputs: pd.DataFrame, train_size: float) -> pd.DataFrame:

    if 0 < train_size < 1:
        # Merge features and labels
        df = pd.merge(left=inputs, right=outputs, how='left', right_index=True, left_index=True)

        # Shuffle for randomization before split
        df = df.sample(frac=1, random_state=42)

        # Split rows
        ix = int(train_size * len(df))
        train_inputs = df.iloc[:ix, :len(inputs.columns)]
        test_inputs = df.iloc[ix:, :len(inputs.columns)]
        train_outputs = df.iloc[:ix, -len(outputs.columns):]
        test_outputs = df.iloc[ix:, -len(outputs.columns):]

        return train_inputs, test_inputs, train_outputs, test_outputs
    else:
        raise ValueError("train_size should be between 0 and 1.")


def evaluate_model(inputs: pd.DataFrame, outputs: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """ Evaluate metrics about the model.

    Args:
        inputs: Art market data to predict.
        outputs: True values.
    
    Side effects:
        Generates `./report.png`
    """
    # Create the estimated prices for each.
    nn = outputs['hammer_price'].notna()
    estimated_price = np.dot(inputs.values, model.x)
    err = (estimated_price[nn] - outputs['hammer_price'][nn])
    prob_lot_sells = stats.norm.cdf(estimated_price - outputs['estimate_low_y'], scale=err.std())

    # -- Plot some useful metrics --

    # Log Model errors.
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20, 12)

    axes[0,0].hist(err, bins=50)
    axes[0,0].set_title('Model Errors')

    # Show the coefficients of the model.
    axes[0,1].bar(x=inputs.columns, height=model.x)
    axes[0,1].set_xticklabels(inputs.columns, rotation=45)
    axes[0,1].set_title('Model Coefficients')
    
    # Normal prob plot of errors - is our Guassian assumption for errors accurate?
    stats.probplot(err, plot=axes[1,0])

    # Show a confusion matrix for categorization of whether a lot will sell.
    # This is a bit of a hack, but its still useful to see.
    conf_matrix = metrics.confusion_matrix(outputs['hammer_price'].notna(), estimated_price > outputs['estimate_low_y'])
    sn.heatmap(pd.DataFrame(conf_matrix, index=['!Sale', 'Sale'], columns=['E[!Sale]', 'E[Sale]']), ax=axes[1,1], 
                            annot=True, annot_kws={"size": 6}, fmt='.6g')

    # Show a histogram of the probability that a lot sells.
    axes[0,2].hist(prob_lot_sells, bins=50)
    axes[0,2].set_title('Probability Distribution for Does Lot Sell Estimate')

    # Scatter plot to show prediction vs reality.
    axes[1,2].scatter(estimated_price[nn], outputs['hammer_price'][nn])
    rsquared = np.power(np.corrcoef(estimated_price[nn], outputs['hammer_price'][nn])[0,1], 2)
    axes[1,2].set_title('Model Estimate vs Realized : {0:.4f} R squared'.format(rsquared))

    # plt.figure(figsize=(20, 12), dpi=320, facecolor='w', edgecolor='k')
    plt.savefig('report.png', bbox_inches='tight', dpi=320)


if __name__ == "__main__":
    # Load data.
    df = pd.read_csv('artists/picasso.csv') 

    # Preprocess.
    inputs, outputs = preprocess_inputs(df, zscore=True)

    # Split train and test sets
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, train_size=0.80)

    # Train model.
    model = train_model(train_inputs, train_outputs)

    # Evaluate model.
    evaluate_model(test_inputs, test_outputs, model)
