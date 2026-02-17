# استيراد المكتبات المطلوبة
# pandas: لمعالجة البيانات وقراءة ملفات CSV
# matplotlib: لعرض الرسوم البيانية
# numpy: للعمليات العددية
# train_test_split: لتقسيم البيانات إلى تدريب واختبار
# LinearRegression: نموذج الانحدار الخطي
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# تحميل البيانات (العلاقة بين الراتب وسنوات الخبرة)
dataset = pd.read_csv('Salary_Data.csv')

# استخراج المتغير المستقل (Years of Experience)
# اختيار جميع الصفوف وكل الأعمدة ما عدا العمود الأخير
X = dataset.iloc[:,:-1].values

# استخراج المتغير التابع (Salary)
# اختيار جميع الصفوف والعمود الأخير فقط
y = dataset.iloc[:,-1].values


# تقسيم البيانات إلى:
# 80% تدريب و 20% اختبار
# random_state لضمان نفس النتيجة في كل تشغيل
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# إنشاء نموذج الانحدار الخطي
regressor = LinearRegression()

# تدريب النموذج على بيانات التدريب
regressor.fit(X_train, y_train)


# التنبؤ بقيم الرواتب لبيانات الاختبار
y_pred = regressor.predict(X_test)



# ---------------------------------
# عرض النتائج - بيانات التدريب
# ---------------------------------
# رسم نقاط بيانات التدريب الفعلية
# رسم خط الانحدار الناتج عن النموذج
# (حالياً هذا الجزء معطّل بالتعليق)

# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()



# ---------------------------------
# عرض النتائج - بيانات الاختبار
# ---------------------------------

# رسم نقاط بيانات الاختبار الفعلية
plt.scatter(X_test, y_test, color='red')

# رسم خط الانحدار المعتمد على النموذج المدرّب
plt.plot(X_train, regressor.predict(X_train), color='blue')

# إضافة عنوان ومحاور للرسم
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# عرض الرسم البياني
plt.show()
