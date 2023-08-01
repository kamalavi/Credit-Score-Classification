#Data visualization of categorical variables
import data_cleaning as dc

#specify path for data
path = r"00. Data\\train.csv"

df, col_float, col_string = dc.data_cleaning(path)

def data_visualization():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, len(col_string),squeeze=False, figsize=(20,7))
    cm = plt.get_cmap('gist_rainbow')

    for i in range(len(col_string)):
        var = col_string[i]
        num_colors = len(df[var].value_counts())
        
        colors = [cm(1.*j/num_colors) for j in range(num_colors)]
        ax[0,i].bar(df[var].unique(), (df[var].value_counts()), color = colors)
        ax[0,i].tick_params(axis='x', labelrotation = 90)
        ax[0,i].set_xlabel(var.replace("_"," "), fontsize = 12)
        
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.show()

    #data visualization for numeric variables
    fig, ax = plt.subplots(5, 3, figsize = (12,20))
    ax = ax.flatten()
    colors = ['b', 'r', 'g']

    for i in range(len(col_float)):
        boxplot_data = {}
        
        for credit_score in df['credit_score'].unique():
            boxplot_data[credit_score] = df[df['credit_score']==credit_score][[col_float[i]]].values.flatten()
            bp_dict = ax[i].boxplot(boxplot_data.values(), labels=boxplot_data.keys(), patch_artist=True, flierprops = dict(marker = 'o',markersize = 3, linestyle = 'none'), whis = 1.5)
        
        bp_dict['boxes'][0].set_facecolor(colors[0])
        bp_dict['boxes'][1].set_facecolor(colors[1])
        bp_dict['boxes'][2].set_facecolor(colors[2])

        bp_dict['medians'][0].set_color('white')
        bp_dict['medians'][1].set_color('white')
        bp_dict['medians'][2].set_color('white')
        
        ax[i].set_xlabel(col_float[i].replace("_"," "), fontsize = 12)
        ax[i].set_ylabel('Feature Value',fontsize = 12)
        ax[i].grid(color = 'white', linestyle = '-', linewidth = 2, alpha = 0.5)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
    plt.show()