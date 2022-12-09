
Output exceeds the size limit. Open the full output data in a text editor
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1955 entries, 1 to 1955
Data columns (total 19 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Age                       1861 non-null   float64
 1   TypeofContact             1945 non-null   object 
 2   CityTier                  1955 non-null   int64  
 3   DurationOfPitch           1853 non-null   float64
 4   Occupation                1955 non-null   object 
 5   Gender                    1955 non-null   object 
 6   NumberOfPersonVisiting    1955 non-null   int64  
 7   NumberOfFollowups         1942 non-null   float64
 8   ProductPitched            1955 non-null   object 
 9   PreferredPropertyStar     1945 non-null   float64
 10  MaritalStatus             1955 non-null   object 
 11  NumberOfTrips             1898 non-null   float64
 12  Passport                  1955 non-null   int64  
 13  PitchSatisfactionScore    1955 non-null   int64  
 14  OwnCar                    1955 non-null   int64  
 15  NumberOfChildrenVisiting  1928 non-null   float64
 16  Designation               1955 non-null   object 
 17  MonthlyIncome             1855 non-null   float64
 18  ProdTaken                 1955 non-null   int64  
dtypes: float64(7), int64(6), object(6)
...
 17  MonthlyIncome             2800 non-null   float64
dtypes: float64(7), int64(5), object(6)
memory usage: 435.4+ KB
None

dtypes: float64(7), int64(5), object(6) 데이터 타입을 확인합니다.
Object 타입인 ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation'] 컬럼은 이후 학습을 위해 
인코딩이 필요합니다.

Output exceeds the size limit. Open the full output data in a text editor
               Age     CityTier  DurationOfPitch  NumberOfPersonVisiting  \
count  1861.000000  1955.000000      1853.000000             1955.000000   
mean     37.462117     1.641432        15.524015                2.922762   
std       9.189948     0.908744         8.150057                0.712276   
min      18.000000     1.000000         5.000000                1.000000   
25%      31.000000     1.000000         9.000000                2.000000   
50%      36.000000     1.000000        14.000000                3.000000   
75%      43.000000     3.000000        20.000000                3.000000   
max      61.000000     3.000000        36.000000                5.000000   

       NumberOfFollowups  PreferredPropertyStar  NumberOfTrips     Passport  \
count        1942.000000            1945.000000    1898.000000  1955.000000   
mean            3.718332               3.568638       3.255532     0.291049   
std             1.004095               0.793196       1.814698     0.454362   
min             1.000000               3.000000       1.000000     0.000000   
25%             3.000000               3.000000       2.000000     0.000000   
50%             4.000000               3.000000       3.000000     0.000000   
75%             4.000000               4.000000       4.000000     1.000000   
max             6.000000               5.000000      19.000000     1.000000   

'''
컬럼 참고용 
id : 샘플 아이디
Age : 나이
TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
Occupation : 직업
Gender : 성별
NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
ProductPitched : 영업 사원이 제시한 상품
PreferredPropertyStar : 선호 호텔 숙박업소 등급
MaritalStatus : 결혼여부
NumberOfTrips : 평균 연간 여행 횟수
Passport : 여권 보유 여부 (0: 없음, 1: 있음)
PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
Designation : (직업의) 직급
MonthlyIncome : 월 급여
ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)
'''
Output exceeds the size limit. Open the full output data in a text editor
Age                          94
TypeofContact                10
CityTier                      0
DurationOfPitch             102
Occupation                    0
Gender                        0
NumberOfPersonVisiting        0
NumberOfFollowups            13
ProductPitched                0
PreferredPropertyStar        10
MaritalStatus                 0
NumberOfTrips                57
Passport                      0
PitchSatisfactionScore        0
OwnCar                        0
NumberOfChildrenVisiting     27
Designation                   0
MonthlyIncome               100
ProdTaken                     0
dtype: int64
Age                         132
TypeofContact                15
CityTier                      0
DurationOfPitch             149
Occupation                    0
...
NumberOfChildrenVisiting     39
Designation                   0
MonthlyIncome               133
dtype: int64

'\n컬럼 참고용 \nid : 샘플 아이디\nAge : 나이\nTypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)\nCityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) \nDurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간\nOccupation : 직업\nGender : 성별\nNumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원\nNumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수\nProductPitched : 영업 사원이 제시한 상품\nPreferredPropertyStar : 선호 호텔 숙박업소 등급\nMaritalStatus : 결혼여부\nNumberOfTrips : 평균 연간 여행 횟수\nPassport : 여권 보유 여부 (0: 없음, 1: 있음)\nPitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도\nOwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)\nNumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수\nDesignation : (직업의) 직급\nMonthlyIncome : 월 급여\nProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)\n'
       PitchSatisfactionScore       OwnCar  NumberOfChildrenVisiting  \
count             1955.000000  1955.000000               1928.000000   
mean                 3.067519     0.619437                  1.213174   
std                  1.372915     0.485649                  0.859450   
min                  1.000000     0.000000                  0.000000   
...
25%     20390.000000     0.000000  
50%     22295.000000     0.000000  
75%     25558.000000     0.000000  
max     98678.000000     1.000000  

train과 test 모두 ['Age','DurationOfPitch','NumberOfFollowups','PreferredPropertyStar',
'NumberOfTrips','NumberOfChildrenVisiting','NumberOfChildrenVisiting']의 컬럼이 결측이 존재합니다.

