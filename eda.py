# %%
import numpy as np
import pandas as pd
import os
import util as ut
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
pd.options.display.max_rows = None
%matplotlib inline
# %%
print(
    f"files and folders: {os.listdir('/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final')}")
print("Subfolders in images folder: ", len(
    list(os.listdir("/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final/images"))))
# %%
total_folders = total_files = 0
folder_info = []
images_names = []
for base, dirs, files in tqdm(os.walk('/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final')):
    for directories in dirs:
        folder_info.append(
            (directories, len(
                os.listdir(
                    os.path.join(
                        base, directories)))))
        total_folders += 1
    for _files in files:
        total_files += 1
        if len(_files.split(".jpg")) == 2:
            images_names.append(_files.split(".jpg")[0])
# %%
print(
    f"Total number of folders: {total_folders}\nTotal number of files: {total_files}")
folder_info_df = pd.DataFrame(folder_info, columns=["folder", "files count"])
folder_info_df.sort_values(["files count"], ascending=False).head()
# %%
print("folder names: ", list(folder_info_df.folder.unique()))
# %%
articles_df = pd.read_csv(
    "/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final/raw_data/articles.csv")
customers_df = pd.read_csv(
    "/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final/raw_data/customers.csv")
sample_submission_df = pd.read_csv(
    "/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final/raw_data/sample_submission.csv")
# %%
transactions_train_df = pd.read_csv(
    "/Users/dmitriiannenkov/Documents/GitHub/Data_Viz_Final/raw_data/transactions_train.csv")
# %%
id_to_index_dict = dict(zip(customers_df['customer_id'], customers_df.index))
index_to_id_dict = dict(zip(customers_df.index, customers_df['customer_id']))
# %%
transactions_train_df['customer_id'] = transactions_train_df['customer_id'].map(
    id_to_index_dict)
customers_df['customer_id'] = customers_df['customer_id'].map(id_to_index_dict)
transactions_train_df['article_id'] = transactions_train_df['article_id'].astype(
    'int32')
# %%
transactions_train_grouped = transactions_train_df.groupby(
    ['customer_id'])['price'].sum().reset_index()
customers_transactions_merged_df = transactions_train_grouped.merge(
    customers_df, on='customer_id', how='left')


# %%
ut.missing_data(articles_df)
# %%
ut.missing_data(customers_df)
# %%
ut.missing_data(transactions_train_df)
# %%
ut.unique_values(articles_df)
# %%
ut.unique_values(customers_df)
# %%
ut.unique_values(transactions_train_df)
# %%
print(
    f"Percent of articles present in transactions: {round(transactions_train_df.article_id.nunique()/articles_df.article_id.nunique(),3)*100}%")
print(
    f"Percent of articles present in transactions: {round(transactions_train_df.customer_id.nunique()/customers_df.customer_id.nunique(),3)*100}%")
# %% Articles data
temp = articles_df.groupby(["product_group_name"])[
    "product_type_name"].nunique()
df = pd.DataFrame({'Product Group': temp.index,
                   'Product Types': temp.values
                   })
df = df.sort_values(['Product Types'], ascending=False)
plt.figure(figsize=(8, 6))
plt.title('Number of Product Types per each Product Group')
sn.set_color_codes("pastel")
s = sn.barplot(x='Product Group', y="Product Types", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Product Types per each Product Group.png')
plt.show()

# %%
ut.show_wordcloud(articles_df["prod_name"], "Wordcloud from product name")

# %%
temp = articles_df.groupby(["product_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Product Group': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(8, 6))
plt.title('Number of Articles per each Product Group')
sn.set_color_codes("pastel")
s = sn.barplot(x='Product Group', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.show()
# %%
f, ax = plt.subplots(figsize=(21, 7))
ax = sn.histplot(data=articles_df, y='garment_group_name',
                 color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.savefig('visualizations/Garment Group Histplot.png')
plt.show()

# %%
temp = articles_df.groupby(["product_type_name"])["article_id"].nunique()
df = pd.DataFrame({'Product Type': temp.index,
                   'Articles': temp.values
                   })
total_types = len(df['Product Type'].unique())
df = df.sort_values(['Articles'], ascending=False)[0:50]
plt.figure(figsize=(16, 6))
plt.title(
    f'Number of Articles per each Product Type (top 50 from total: {total_types})')
sn.set_color_codes("pastel")
s = sn.barplot(x='Product Type', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Product Type.png')
plt.show()
# %%
temp = articles_df.groupby(["department_name"])["article_id"].nunique()
df = pd.DataFrame({'Department Name': temp.index,
                   'Articles': temp.values
                   })
total_depts = len(df['Department Name'].unique())
df = df.sort_values(['Articles'], ascending=False).head(50)
plt.figure(figsize=(16, 6))
plt.title(
    f'Number of Articles per each Department (top 50 from total: {total_depts})')
sn.set_color_codes("pastel")
s = sn.barplot(x='Department Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Department.png')
plt.show()
# %%
temp = articles_df.groupby(["graphical_appearance_name"])[
    "article_id"].nunique()
df = pd.DataFrame({'Graphical Appearance Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False).head(50)
plt.figure(figsize=(16, 6))
plt.title(f'Number of Articles per each Graphical Appearance Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Graphical Appearance Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig(
    'visualizations/Number of Articles per each Graphical Appearance Name.png')
plt.show()
# %%
temp = articles_df.groupby(["index_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Index Group Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(6, 6))
plt.title(f'Number of Articles per each Index Group Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Index Group Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Index Group Name.png')
plt.show()
# %%
temp = articles_df.groupby(["colour_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Colour Group Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(12, 6))
plt.title(f'Number of Articles per each Colour Group Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Colour Group Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Colour Group Name.png')
plt.show()
# %%
temp = articles_df.groupby(["perceived_colour_value_name"])[
    "article_id"].nunique()
df = pd.DataFrame({'Perceived Colour Group Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(6, 6))
plt.title(f'Number of Articles per each Perceived Colour Group Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Perceived Colour Group Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig(
    'visualizations/Number of Articles per each Perceived Colour Group Name.png')
plt.show()
# %%
temp = articles_df.groupby(["perceived_colour_master_name"])[
    "article_id"].nunique()
df = pd.DataFrame({'Perceived Colour Master Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(12, 6))
plt.title(f'Number of Articles per each Perceived Colour Master Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Perceived Colour Master Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig(
    'visualizations/Number of Articles per each Perceived Colour Master Name.png')
plt.show()
# %%
temp = articles_df.groupby(["index_name"])["article_id"].nunique()
df = pd.DataFrame({'Index Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(8, 6))
plt.title(f'Number of Articles per each Index Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Index Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Index Name.png')
plt.show()
# %%
temp = articles_df.groupby(["garment_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Garment Group Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(12, 6))
plt.title(f'Number of Articles per each Garment Group Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Garment Group Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Garment Group Name.png')
plt.show()
# %%
temp = articles_df.groupby(["section_name"])["article_id"].nunique()
df = pd.DataFrame({'Section Name': temp.index,
                   'Articles': temp.values
                   })
df = df.sort_values(['Articles'], ascending=False)
plt.figure(figsize=(16, 6))
plt.title(f'Number of Articles per each Section Name')
sn.set_color_codes("pastel")
s = sn.barplot(x='Section Name', y="Articles", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Articles per each Section Name.png')
plt.show()
# %%
ut.show_wordcloud(articles_df["detail_desc"],
                  "Wordcloud from detailed description of articles")
# %% Customer Data
temp = customers_df.groupby(["age"])["customer_id"].count()
df = pd.DataFrame({'Age': temp.index,
                   'Customers': temp.values
                   })
df = df.sort_values(['Age'], ascending=False)
plt.figure(figsize=(16, 6))
plt.title(f'Number of Customers per each Age')
sn.set_color_codes("pastel")
s = sn.barplot(x='Age', y="Customers", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Customers per each Age.png')
plt.show()
# %%
temp = customers_df.groupby(["fashion_news_frequency"])["customer_id"].count()
df = pd.DataFrame({'Fashion News Frequency': temp.index,
                   'Customers': temp.values
                   })
df = df.sort_values(['Customers'], ascending=False)
plt.figure(figsize=(6, 6))
plt.title(f'Number of Customers per each Fashion News Frequency')
sn.set_color_codes("pastel")
s = sn.barplot(x='Fashion News Frequency', y="Customers", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig(
    'visualizations/Number of Customers per each Fashion News Frequency.png')
plt.show()
# %%
temp = customers_df.groupby(["club_member_status"])["customer_id"].count()
df = pd.DataFrame({'Club Member Status': temp.index,
                   'Customers': temp.values
                   })
df = df.sort_values(['Customers'], ascending=False)
plt.figure(figsize=(6, 6))
plt.title(f'Number of Customers per each Club Member Status')
sn.set_color_codes("pastel")
s = sn.barplot(x='Club Member Status', y="Customers", data=df)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
locs, labels = plt.xticks()
plt.savefig('visualizations/Number of Customers per each Club Member Status.png')
plt.show()
# %% Transactions data
df = transactions_train_df  # .sample(1_000_000)
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sn.kdeplot(np.log(df.loc[df["sales_channel_id"] == 1].price.value_counts()))
sn.kdeplot(np.log(df.loc[df["sales_channel_id"] == 2].price.value_counts()))
ax.legend(labels=['Sales channel 1', 'Sales channel 1'])
plt.title("Logaritmic distribution of price frequency in transactions, grouped per sales channel")
plt.savefig('visualizations/Logaritmic distribution of price frequency in transactions, grouped per sales channel.png')
plt.show()
# %%
df = transactions_train_df.groupby(
    ["t_dat"])["article_id"].count().reset_index()
df["t_dat"] = df["t_dat"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.columns = ["Date", "Transactions"]
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
plt.plot(df["Date"], df["Transactions"], color="Darkgreen")
plt.xlabel("Date")
plt.ylabel("Transactions")
plt.title(
    f"Transactions per day")
plt.savefig('visualizations/Transactions per day.png')
plt.show()
# %%
df = transactions_train_df.groupby(
    ["t_dat", "sales_channel_id"])["article_id"].count().reset_index()
df["t_dat"] = df["t_dat"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.columns = ["Date", "Sales Channel Id", "Transactions"]
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
g1 = ax.plot(df.loc[df["Sales Channel Id"] == 1,
                    "Date"],
             df.loc[df["Sales Channel Id"] == 1,
                    "Transactions"],
             label="Sales Channel 1",
             color="Darkblue")
g2 = ax.plot(df.loc[df["Sales Channel Id"] == 2,
                    "Date"],
             df.loc[df["Sales Channel Id"] == 2,
                    "Transactions"],
             label="Sales Channel 2",
             color="Magenta")
plt.xlabel("Date")
plt.ylabel("Transactions")
ax.legend()
plt.title(
    f"Transactions per day, grouped by Sales Channel")
plt.savefig('visualizations/Transactions per day, grouped by Sales Channel.png')
plt.show()
# %%
df = transactions_train_df.groupby(["t_dat", "sales_channel_id"])[
    "article_id"].nunique().reset_index()
df["t_dat"] = df["t_dat"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.columns = ["Date", "Sales Channel Id", "Unique Articles"]
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
g1 = ax.plot(df.loc[df["Sales Channel Id"] == 1,
                    "Date"],
             df.loc[df["Sales Channel Id"] == 1,
                    "Unique Articles"],
             label="Sales Channel 1",
             color="Blue")
g2 = ax.plot(df.loc[df["Sales Channel Id"] == 2,
                    "Date"],
             df.loc[df["Sales Channel Id"] == 2,
                    "Unique Articles"],
             label="Sales Channel 2",
             color="Green")
plt.xlabel("Date")
plt.ylabel("Unique Articles / Day")
ax.legend()
plt.title(f"Unique articles per day, grouped by Sales Channel")
plt.savefig(
    'visualizations/Unique articles per day, grouped by Sales Channel.png')
plt.show()
# %%
transactions_train_df.t_dat = pd.to_datetime(transactions_train_df.t_dat)
transactions_train_df['year'] = (
    transactions_train_df.t_dat.dt.year - 2000).astype('int8')
transactions_train_df['month'] = transactions_train_df.t_dat.dt.month.astype(
    'int8')
transactions_train_df['day'] = transactions_train_df.t_dat.dt.day.astype(
    'int8')
# %%
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(20, 5))
ax = sn.boxplot(data=transactions_train_df, x='price', color='orange')
ax.set_xlabel('Price outliers')
plt.savefig('visualizations/Price outliers.png')
plt.show()
# %%
articles_for_merge = articles_df[['article_id',
                                  'prod_name',
                                  'product_type_name',
                                  'product_group_name',
                                  'index_name']]
articles_for_merge = transactions_train_df[['customer_id', 'article_id', 'price', 't_dat']].merge(
    articles_for_merge, on='article_id', how='left')
# %%
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(30, 18))
ax = sn.boxplot(data=articles_for_merge, x='price', y='product_group_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
plt.savefig('visualizations/Price outliers per index name.png')
plt.show()
# %%
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(30, 20))
_ = articles_for_merge[articles_for_merge['product_group_name']
                       == 'Accessories']
ax = sn.boxplot(data=_, x='price', y='product_type_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
del _
plt.savefig('visualizations/Price outliers per index name, Accessories.png')
plt.show()
# %%
articles_index = articles_for_merge[[
    'index_name', 'price']].groupby('index_name').mean()
# %%
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(24, 8))
ax = sn.barplot(x=articles_index.price, y=articles_index.index,
                color='orange', alpha=0.8)
ax.set_xlabel('Price by index')
ax.set_ylabel('Index')
plt.savefig('visualizations/Price by index.png')
plt.show()
# %%
articles_index = articles_for_merge[['product_group_name', 'price']].groupby(
    'product_group_name').mean()
sn.set_style("darkgrid")
f, ax = plt.subplots(figsize=(24, 8))
ax = sn.barplot(x=articles_index.price, y=articles_index.index,
                color='orange', alpha=0.8)
ax.set_xlabel('Price by product group')
ax.set_ylabel('Product group')
plt.savefig('visualizations/Price by product group.png')
plt.show()
# %%
product_list = ['Shoes', 'Garment Full body', 'Bags',
                'Garment Lower body', 'Underwear/nightwear']
colors = ['cadetblue', 'orange',
          'mediumspringgreen', 'tomato', 'lightseagreen']
k = 0
f, ax = plt.subplots(3, 2, figsize=(30, 15))
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
plt.savefig('visualizations/Mean price of producrs in time.png')
plt.show()

# %% Images Data
image_name_df = pd.DataFrame(images_names, columns=["image_name"])
image_name_df["article_id"] = image_name_df["image_name"].apply(lambda x: int(x[1:]))
# %%
image_article_df = articles_df[["article_id",
                                "product_code",
                                "product_group_name",
                                "product_type_name"]].merge(image_name_df,
                                                            on=["article_id"],
                                                            how="left")
print(image_article_df.shape)
image_article_df.head()
# %%
article_no_image_df = image_article_df.loc[image_article_df.image_name.isna()]
print(article_no_image_df.shape)
article_no_image_df.head()
# %%
print("Product codes with some missing images: ",
      article_no_image_df.product_code.nunique())
print("Product groups with some missing images: ", list(
    article_no_image_df.product_group_name.unique()))
# %%
print(image_article_df.product_group_name.unique())
# %%
ut.plot_image_samples(image_article_df, "Garment Lower body", 4, 2)

# %%
ut.plot_image_samples(image_article_df, "Stationery", 4, 1)
# %%
ut.plot_image_samples(image_article_df, "Fun", 2, 1)
# %%
ut.plot_image_samples(image_article_df, "Accessories", 4, 1)
# %%
ut.plot_image_samples(image_article_df, "Swimwear", 4, 2)
# %%
ut.plot_image_samples(image_article_df, "Furniture", 4, 2)
# %%
ut.plot_image_samples(image_article_df, "Cosmetic", 4, 1)
# %%
ut.plot_image_samples(image_article_df, "Bags", 4, 3)
# %% Training Data
recent_transactions_train_df = transactions_train_df.sort_values(
    ["customer_id", "t_dat"], ascending=False)

# %%
last_date = recent_transactions_train_df.t_dat.max()
print(last_date)
print(
    recent_transactions_train_df.loc[recent_transactions_train_df.t_dat == last_date].shape[1])
# %%
most_frequent_articles = list(
    recent_transactions_train_df.loc[recent_transactions_train_df.t_dat == last_date].article_id.value_counts()[0:12].index)
art_list = []
for art in most_frequent_articles:
    art = "0" + str(art)
    art_list.append(art)
art_str = " ".join(art_list)
print("Frequent articles bought recently: ", art_str)
# %%
recent_transactions_train_df.loc[recent_transactions_train_df.t_dat == last_date].head(
)
# %%
import sweetviz as sv
# %%
articles_report = sv.analyze(articles_df)
customers_report = sv.analyze(customers_df)
transactions_train_report = sv.analyze(transactions_train_df)
articles_index_report = sv.analyze(articles_for_merge)
articles_for_merge_report = sv.analyze(articles_for_merge)
# %%
articles_report.show_html('sweetviz_reports/articles_report.html')
customers_report.show_html('sweetviz_reports/customers_report.html')
transactions_train_report.show_html('sweetviz_reports/transactions_train.html')
articles_index_report.show_html('sweetviz_reports/articles_index_report.html')
articles_for_merge_report.show_html(
    'sweetviz_reports/articles_for_merge_report.html')