# %%
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
print('Importing packages...')

pd.options.display.max_rows = None
print('Packages imported!')

# %matplotlib inline
# %%
print('Loading data...')
articles_df = pd.read_csv('articles.csv')
customers_df = pd.read_csv('customers.csv')
transactions_train_df = pd.read_csv('transactions_train.csv')
print('Data loaded!')
# %%
print('Data exploration...')
print('Creating customer_id_to_index_dict...')
id_to_index_dict = dict(zip(customers_df['customer_id'], customers_df.index))
index_to_id_dict = dict(zip(customers_df.index, customers_df['customer_id']))
# %% for memory efficiency
print('Optimizing data...')
transactions_train_df['customer_id'] = transactions_train_df['customer_id'].map(
    id_to_index_dict)
customers_df['customer_id'] = customers_df['customer_id'].map(id_to_index_dict)
transactions_train_df['article_id'] = transactions_train_df['article_id'].astype(
    'int32')
transactions_train_df.t_dat = pd.to_datetime(transactions_train_df.t_dat)
transactions_train_df['year'] = (
    transactions_train_df.t_dat.dt.year - 2000).astype('int8')
transactions_train_df['month'] = transactions_train_df.t_dat.dt.month.astype(
    'int8')
transactions_train_df['day'] = transactions_train_df.t_dat.dt.day.astype(
    'int8')

# %%
print('Merging data...')
transactions_train_grouped = transactions_train_df.groupby(
    ['customer_id'])['price'].sum().reset_index()
customers_transactions_merged_df = transactions_train_grouped.merge(
    customers_df, on='customer_id', how='left')
# %%
print('Plotting index_name histogram...')
f, ax = plt.subplots(figsize=(21, 7))
ax = sn.histplot(data=articles_df, y='index_name', color='orange')
ax.set_xlabel('count by index name')
ax.set_ylabel('index name')
# plt.savefig('visualizations/index_name_hist.png')
plt.show()

# %%
print('Plotting garment_group_name_by_index_group_name histogram...')
f, ax = plt.subplots(figsize=(21, 7))
ax = sn.histplot(data=articles_df, y='garment_group_name',
                 color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
# plt.savefig('visualizations/garment_group_name_by_index_group_name.png')
plt.show()

# %%
articles_df.groupby(['index_group_name', 'index_name']).count()['article_id']

# %%
articles_df.groupby(['product_group_name', 'product_type_name']).count()[
    'article_id']

# %%
print('Printing uniqie articles...')
for col in articles_df.columns:
    if 'no' not in col and 'code' not in col and 'id' not in col:
        un_n = articles_df[col].nunique()
        print(f'№ of unique {col}: {un_n}')

# %%
print('Plotting customers_age histogram...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.histplot(data=customers_df, x='age', bins=50, color='orange')
ax.set_xlabel('Distribution of the customers age')
plt.savefig('visualizations/customers_age_distribution.png')
plt.show()

# %%
print('Plotting club_membership_status histogram...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.histplot(data=customers_df, x='club_member_status', color='orange')
ax.set_xlabel('Distribution of club member status')
plt.savefig('visualizations/club_member_status.png')
plt.show()

# %%
print('Plotting price outliers...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.boxplot(data=transactions_train_df, x='price', color='orange')
ax.set_xlabel('Price outliers')
plt.savefig('visualizations/price_outliers.png')
plt.show()

# %%
print('Left join (Merge) articles_df and transactions_train_df...')
articles_for_merge = articles_df[['article_id',
                                  'prod_name',
                                  'product_type_name',
                                  'product_group_name',
                                  'index_name']]
articles_for_merge = transactions_train_df[['customer_id', 'article_id', 'price', 't_dat']].merge(
    articles_for_merge, on='article_id', how='left')

# %%
print('Plotting price_by_index_name boxplot...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(30, 18))
ax = sn.boxplot(data=articles_for_merge, x='price', y='product_group_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
plt.savefig('visualizations/price_by_index_name.png')
plt.show()

# %%
print('Plotting price_outliers_by_index_name boxplot...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(25, 18))
_ = articles_for_merge[articles_for_merge['product_group_name']
                       == 'Accessories']
ax = sn.boxplot(data=_, x='price', y='product_type_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
del _
plt.savefig('visualizations/price_outliers_by_index_name.png')
plt.show()

# %%
print('Grouping articles_for_merge by index_name...')
articles_index = articles_for_merge[[
    'index_name', 'price']].groupby('index_name').mean()

# %%
print('Plotting price_by_index barplot...')
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.barplot(x=articles_index.price, y=articles_index.index,
                color='orange', alpha=0.8)
ax.set_xlabel('Price by index')
ax.set_ylabel('Index')
plt.savefig('visualizations/price_by_index.png')
plt.show()

# %%
print('Plotting price_by_product_group boxplot...')
articles_index = articles_for_merge[['product_group_name', 'price']].groupby(
    'product_group_name').mean()
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.barplot(x=articles_index.price, y=articles_index.index,
                color='orange', alpha=0.8)
ax.set_xlabel('Price by product group')
ax.set_ylabel('Product group')
plt.savefig('visualizations/price_by_product_group.png')
plt.show()

# %%
print('Plotting price_by_product_group_time plots...')
product_list = ['Shoes', 'Garment Full body', 'Bags',
                'Garment Lower body', 'Underwear/nightwear']
colors = ['cadetblue', 'orange',
          'mediumspringgreen', 'tomato', 'lightseagreen']
k = 0
f, ax = plt.subplots(3, 2, figsize=(20, 15))
for i in range(3):
    for j in range(2):
        try:
            product = product_list[k]
            articles_for_merge_product = articles_for_merge[
                articles_for_merge.product_group_name == product_list[k]]
            series_mean = articles_for_merge_product[['t_dat',
                                                      'price']].groupby(
                pd.Grouper(key="t_dat", freq='M')).mean().fillna(0)
            series_std = articles_for_merge_product[['t_dat', 'price']].groupby(
                pd.Grouper(key="t_dat", freq='M')).std().fillna(0)
            ax[i, j].plot(series_mean, linewidth=4, color=colors[k])
            ax[i,
               j].fill_between(series_mean.index,
                               (series_mean.values - 2 *
                                series_std.values).ravel(),
                               (series_mean.values + 2 *
                                series_std.values).ravel(),
                               color=colors[k],
                               alpha=.1)
            ax[i, j].set_title(f'Mean {product_list[k]} price in time')
            ax[i, j].set_xlabel('month')
            ax[i, j].set_xlabel(f'{product_list[k]}')
            k += 1
        except IndexError:
            ax[i, j].set_visible(False)
plt.savefig('visualizations/price_by_product_group_time.png')
plt.show()

print('Data exploration finished!')
# %%
print('Creating sweetviz reports...')
articles_report = sv.analyze(articles_df)
customers_report = sv.analyze(customers_df)
transactions_train_report = sv.analyze(transactions_train_df)
articles_index_report = sv.analyze(articles_for_merge)
articles_for_merge_report = sv.analyze(articles_for_merge)

articles_report.show_html('sweetviz_reports/articles_report.html')
customers_report.show_html('sweetviz_reports/customers_report.html')
transactions_train_report.show_html('sweetviz_reports/transactions_train.html')
articles_index_report.show_html('sweetviz_reports/articles_index_report.html')
articles_for_merge_report.show_html(
    'sweetviz_reports/articles_for_merge_report.html')
print('Reports created!')

print('Done!')
