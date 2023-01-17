import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()


# Calculate information value
def calc_iv(df, feature, target, pr=False):
    lst = []
    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    print('This variable\'s IV is:',iv)
    print(df[feature].value_counts())
    return iv, data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




data = pd.read_csv("application_record.csv", encoding = 'utf-8')
record = pd.read_csv("credit_record.csv", encoding = 'utf-8')






plt.rcParams['figure.facecolor'] = 'white'

# find all users' account open month.
begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'})

record['target'] = 0
record['target'][record['STATUS'] =='2']= 1
record['target'][record['STATUS'] =='3']= 1
record['target'][record['STATUS'] =='4']= 1
record['target'][record['STATUS'] =='5']= 1
st=pd.DataFrame(record.groupby(["ID"])["target"].agg(max))

# Add target column
data_table=pd.merge(data,st,how="inner",on="ID")

# inputation using mode
data_table['OCCUPATION_TYPE'].fillna(data_table['OCCUPATION_TYPE'].mode()[0], inplace=True)
data_table['target'].fillna(data_table['target'].mean(), inplace=True)

data_table.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Reality',
                         'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',
                         'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',
                        'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',
                         'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',
                         'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',
                        'OCCUPATION_TYPE':'occyp'
                        },inplace=True)

print(data_table)
# Print out number of missing values for each feature
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{data_table.isnull().sum()}")

print(data_table.shape)
print(data_table['target'].value_counts())

data_table['Gender'] = data_table['Gender'].replace(['F','M'],[0,1])
print(data_table['Gender'].value_counts())

data_table['Car'] = data_table['Car'].replace(['N','Y'],[0,1])
print(data_table['Car'].value_counts())

data_table['Reality'] = data_table['Reality'].replace(['N','Y'],[0,1])
print(data_table['Reality'].value_counts())

data_table['phone']=data_table['phone'].astype(str)
print(data_table['phone'].value_counts(normalize=True,sort=False))
data_table['phone'].fillna(data_table['phone'].mode()[0], inplace=True)
data_table['phone'] = ord_enc.fit_transform(data_table[["phone"]])

print(data_table['email'].value_counts(normalize=True,sort=False))
data_table['email']=data_table['email'].astype(str)
data_table['email'] = ord_enc.fit_transform(data_table[["email"]])

data_table['wkphone']=data_table['wkphone'].astype(str)
data_table['wkphone'].fillna(data_table['wkphone'].mode()[0], inplace=True)
data_table['wkphone'] = ord_enc.fit_transform(data_table[["wkphone"]])

data_table['inc']=data_table['inc'].astype(object)
data_table['inc'] = data_table['inc']//10000

data_table['Age']=-(data_table['DAYS_BIRTH'])//365

data_table['worktm']=-(data_table['DAYS_EMPLOYED'])//365
data_table['worktm'][data_table['worktm']<0] = np.nan # replace by na
data_table['worktm'].fillna(data_table['worktm'].mean(),inplace=True) #replace na by mean

print(data_table['inctp'].value_counts(normalize=True,sort=False))
data_table.loc[data_table['inctp']=='Pensioner','inctp']='State servant'
data_table.loc[data_table['inctp']=='Student','inctp']='State servant'
data_table['inctp'] = ord_enc.fit_transform(data_table[["inctp"]])

data_table.loc[(data_table['occyp']=='Cleaning staff') | (data_table['occyp']=='Cooking staff') | (data_table['occyp']=='Drivers') | (data_table['occyp']=='Laborers') | (data_table['occyp']=='Low-skill Laborers') | (data_table['occyp']=='Security staff') | (data_table['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
data_table.loc[(data_table['occyp']=='Accountants') | (data_table['occyp']=='Core staff') | (data_table['occyp']=='HR staff') | (data_table['occyp']=='Medicine staff') | (data_table['occyp']=='Private service staff') | (data_table['occyp']=='Realty agents') | (data_table['occyp']=='Sales staff') | (data_table['occyp']=='Secretaries'),'occyp']='officewk'
data_table.loc[(data_table['occyp']=='Managers') | (data_table['occyp']=='High skill tech staff') | (data_table['occyp']=='IT staff'),'occyp']='hightecwk'
print(data_table['occyp'].value_counts())

data_table['occyp'] = ord_enc.fit_transform(data_table[["occyp"]])
data_table['houtp'] = ord_enc.fit_transform(data_table[["houtp"]])
data_table['edutp'] = ord_enc.fit_transform(data_table[["edutp"]])
data_table['famtp'].value_counts(normalize=True,sort=False)
data_table['famtp'] = ord_enc.fit_transform(data_table[["famtp"]])

ivtable=pd.DataFrame(data_table.columns,columns=['variable'])
ivtable['IV']=None
iv, data = calc_iv(data_table,'Gender','target')
ivtable.loc[ivtable['variable']=='Gender','IV']=iv
iv, data=calc_iv(data_table,'Car','target')
ivtable.loc[ivtable['variable']=='Car','IV']=iv
iv, data=calc_iv(data_table,'Reality','target')
ivtable.loc[ivtable['variable']=='Reality','IV']=iv
iv, data=calc_iv(data_table,'phone','target')
ivtable.loc[ivtable['variable']=='phone','IV']=iv
iv, data=calc_iv(data_table,'email','target')
ivtable.loc[ivtable['variable']=='email','IV']=iv
iv, data = calc_iv(data_table,'wkphone','target')
ivtable.loc[ivtable['variable']=='wkphone','IV']=iv
iv, data=calc_iv(data_table,'ChldNo','target')
ivtable.loc[ivtable['variable']=='ChldNo','IV']=iv
iv, data = calc_iv(data_table,'Age','target')
ivtable.loc[ivtable['variable']=='Age','IV'] = iv
iv, data=calc_iv(data_table,'inctp','target')
ivtable.loc[ivtable['variable']=='inctp','IV']=iv
iv, data=calc_iv(data_table,'occyp','target')
ivtable.loc[ivtable['variable']=='occyp','IV']=iv
iv, data=calc_iv(data_table,'houtp','target')
ivtable.loc[ivtable['variable']=='houtp','IV']=iv
iv, data=calc_iv(data_table,'famtp','target')
ivtable.loc[ivtable['variable']=='famtp','IV']=iv
iv, data = calc_iv(data_table,'inc','target')
ivtable.loc[ivtable['variable']=='inc','IV']=iv
ivtable = ivtable.dropna()
print(ivtable)
print(ivtable['IV'].sum())

Y = data_table['target']
X = data_table[['Gender','Car','Reality', 'ChldNo','inctp',
              'inc', 'famtp', 'houtp', 'wkphone',
       'phone','email', 'occyp', 'Age']]

Y = Y.astype(int)
X_balance,Y_balance = SMOTE().fit_resample(X,Y)
X_balance = pd.DataFrame(X_balance, columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance,
                                                    stratify=Y_balance, test_size=0.3,
                                                    random_state = 10086)

class_names = ['0','1']

model = DecisionTreeClassifier(max_depth=12,
                    min_samples_split=8,
                    random_state=1024)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))

plot_confusion_matrix(confusion_matrix(y_test,y_predict),
    classes=class_names, normalize = True,
    title='Normalized Confusion Matrix: Decision Tree')


model = RandomForestClassifier(n_estimators=250,
    max_depth=12,
    min_samples_leaf=16
)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))

plot_confusion_matrix(confusion_matrix(y_test,y_predict),
    classes=class_names, normalize = True,
    title='Normalized Confusion Matrix: Ramdom Forests')



model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
plot_confusion_matrix(confusion_matrix(y_test,y_predict),
    classes=class_names, normalize = True,
    title='Normalized Confusion Matrix: KNeighbors')
    

from tkinter import *
from tkinter.ttk import *

listOfInfos = ['Gender','Presence of the Car','Presence of the Reality', 'Number of children','Income type', 'Income', 'Marital status', 'Housing type', 'Presence of the work phone', 'Presence of the mobile phone','Presence of the email', 'Ocupation type', 'Age']

master = Tk(className='Prediction of Loan Eligibility')
master.geometry("600x650")
master.configure(background='gray39')
master.grid_columnconfigure((0,1), weight=1)
header=Label(master,text="AI FINAL PROJECT",font='Courier 35 bold')
n= header.configure(background='gray39')
header.configure(foreground='aquamarine')
header.grid(row=0, columnspan=2)
listOfEntries = []
for i in range(len(listOfInfos)):
    l = Label(master, text=listOfInfos[i], font='Courier 15 bold')
    l.configure(foreground='mediumaquamarine')
    l.configure(background='gray39')
    i=i+1
    l.grid(row=i, column=0)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('F', 'M')
cb.grid(row=1, column=1, pady=10)
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Y', 'N')
cb.grid(row=2, column=1, pady=10)
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Y', 'N')
cb.grid(row=3, column=1, pady=10)
listOfEntries.append(cb)

e = Entry(master, width = 20)
e.grid(row=4, column=1, pady=10)
listOfEntries.append(e)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student')
cb.grid(row=5, column=1, pady=10)
listOfEntries.append(cb)

e = Entry(master, width = 20)
e.grid(row=6, column=1, pady=10)
listOfEntries.append(e)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Civil marriage', 'Married', 'Single / not married', 'Separated', 'Widow')
cb.grid(row=7, column=1, pady=10)#, sticky="ew")
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Rented apartment', 'House / apartment', 'Municipal apartment', 'With parents', 'Co-op apartment', 'Office apartment')
cb.grid(row=8, column=1, pady=10)#, sticky="ew")
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = (1, 0)
cb.grid(row=9, column=1, pady=10)
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = (1, 0)
cb.grid(row=10, column=1, pady=10)
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = (1, 0)
cb.grid(row=11, column=1, pady=10)
listOfEntries.append(cb)

cbVal = StringVar()
cb = Combobox(master, textvariable=cbVal)
cb['values'] = ('Laborers', 'Security staff', 'Sales staff', 'Accountants', 'Managers', 'Drivers', 'Core staff', 'High skill tech staff', 'Cleaning staff', 'Private service staff', 'Cooking staff', 'Low-skill Laborers', 'Medicine staff', 'Secretaries', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff')
cb.grid(row=12, column=1, pady=10)
listOfEntries.append(cb)

e = Entry(master, width = 20)
e.grid(row=13, column=1, pady=10)
listOfEntries.append(e)

def callback():
    listOfData = []
    for e in listOfEntries:
        listOfData.append(e.get())
        e.delete(0, 'end')
    inf = [{'Gender': listOfData[0], 'Car': listOfData[1], 'Reality': listOfData[2], 'ChldNo': listOfData[3], 'inctp': listOfData[4],'inc': listOfData[5], 'famtp': listOfData[6], 'houtp': listOfData[7], 'wkphone': listOfData[8], 'phone': listOfData[9], 'email': listOfData[10], 'occyp': listOfData[11], 'Age': listOfData[12]}]
    print(inf)
    myPredict(inf)

def myPredict(inf):
    infdf = pd.DataFrame(inf)
    infdf['inc'] = infdf['inc'].astype(int)
    infdf['inc'] = infdf['inc']//10000
    infdf.loc[infdf['inctp']=='Pensioner','inctp']='State servant'
    infdf.loc[infdf['inctp']=='Student','inctp']='State servant'
    infdf['inctp'] = ord_enc.fit_transform(infdf[["inctp"]])
    infdf.loc[(infdf['occyp']=='Cleaning staff') | (infdf['occyp']=='Cooking staff') | (infdf['occyp']=='Drivers') | (infdf['occyp']=='Laborers') | (infdf['occyp']=='Low-skill Laborers') | (infdf['occyp']=='Security staff') | (infdf['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
    infdf.loc[(infdf['occyp']=='Accountants') | (infdf['occyp']=='Core staff') | (infdf['occyp']=='HR staff') | (infdf['occyp']=='Medicine staff') | (infdf['occyp']=='Private service staff') | (infdf['occyp']=='Realty agents') | (infdf['occyp']=='Sales staff') | (infdf['occyp']=='Secretaries'),'occyp']='officewk'
    infdf.loc[(infdf['occyp']=='Managers') | (infdf['occyp']=='High skill tech staff') | (infdf['occyp']=='IT staff'),'occyp']='hightecwk'
    infdf['occyp'] = ord_enc.fit_transform(infdf[["occyp"]])
    infdf['houtp'] = ord_enc.fit_transform(infdf[["houtp"]])
    infdf['famtp'] = ord_enc.fit_transform(infdf[["famtp"]])
    infdf['phone'] = ord_enc.fit_transform(infdf[["phone"]])
    infdf['email'] = ord_enc.fit_transform(infdf[["email"]])
    infdf['wkphone'] = ord_enc.fit_transform(infdf[["wkphone"]])
    infdf['Gender'] = infdf['Gender'].replace(['F','M'],[0,1])
    infdf['Car'] = infdf['Car'].replace(['N','Y'],[0,1])
    infdf['Reality'] = infdf['Reality'].replace(['N','Y'],[0,1])
    infdf['Car'] = infdf['Car'].astype(int)
    infdf['Reality'] = infdf['Reality'].astype(int)
    infdf['ChldNo'] = infdf['ChldNo'].astype(int)
    infdf['Age'] = infdf['Age'].astype(int)
    print(infdf)
    out = model.predict(infdf)
    res = 'Not Approved'
    if (out[0] == 1):
        res = 'Approved' 
    ans = Label(master, text=res,  font='Courier 15 bold')
    ans.configure(foreground='red3')
    ans.configure(background='black')
    ans.grid(row=15, columnspan=2)
    print(out)

b = Button(master, text = "OK", width = 20, command = callback)
b.grid(row=14,columnspan=2,pady=5)

mainloop()
