# DACON 여행상품 분류예측

### ** 데이터 분석 **

=======
## 데이터 확인
![image](https://user-images.githubusercontent.com/107663853/206635360-f0a3d1d1-801b-4b29-ba9b-980d8636a764.png)

![image](https://user-images.githubusercontent.com/107663853/206635536-34043159-13f5-41ac-a916-24d15cfcdbfe.png)


dtypes: float64(7), int64(5), object(6) 데이터 타입을 확인합니다. 

Object 타입인 ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation'] 

컬럼은 이후 학습을 위해 인코딩이 필요합니다.

### ** 데이터 분석 **

#### print(train_set.describe())

![image](https://user-images.githubusercontent.com/107663853/206635725-809f6f80-bfab-499a-a734-89410a2e5ca8.png)

##### 결측을 확인합니다.
![image](https://user-images.githubusercontent.com/107663853/206635874-a14e0578-f176-4510-be25-176d67279d54.png)

![image](https://user-images.githubusercontent.com/107663853/206636058-8ccdf8cd-5057-4910-bfb3-169df3120181.png)

train과 test 모두 
['Age','DurationOfPitch','NumberOfFollowups','PreferredPropertyStar', 'NumberOfTrips','NumberOfChildrenVisiting','NumberOfChildrenVisiting']의 
컬럼이 결측이 존재합니다.

#### print(train_set.describe(include=['O']))

![image](https://user-images.githubusercontent.com/107663853/206636320-30909519-cda7-4ec7-b92a-f4d5dfbf2219.png)

![image](https://user-images.githubusercontent.com/107663853/206636396-26f0ef39-75f8-430c-a6e5-48f8e5329187.png)


나이의 결측은 직급별 평균의 나이로 채웁니다.직급과 직업 컬럼으로 평균으로 비교해보았습니다만 
직업별 평균으로 값을 대체하기에는 각 값별 편차가 적어 적절하지 않습니다.직급별 평균은 편차가 크기 때문에 값을 대체하기에 적절합니다.


#### 결측치 처리
![image](https://user-images.githubusercontent.com/107663853/206636575-8eac86ce-c0ca-4a98-82d8-adbc95b72275.png)

![image](https://user-images.githubusercontent.com/107663853/206636782-4c5710d3-2b58-4fd6-bce8-83a492e0d8e9.png)

![image](https://user-images.githubusercontent.com/107663853/206637112-5ad3420e-cda4-474e-92d1-dd7ad56ed896.png)

![image](https://user-images.githubusercontent.com/107663853/206637180-fa7a24c6-b3ed-4d5c-aaf5-d7830f227b27.png)

![image](https://user-images.githubusercontent.com/107663853/206637268-f42f30d8-e932-408f-81d2-c8196d3cb636.png)

![image](https://user-images.githubusercontent.com/107663853/206637373-c9d3ef4b-d34d-4fe6-b6de-6ab5caaae037.png)

![image](https://user-images.githubusercontent.com/107663853/206637474-3aa6934e-6390-4bff-8e95-a510ccb80fbe.png)

#### Feature 선택

![image](https://user-images.githubusercontent.com/107663853/206637546-69818384-5064-4f7c-b74d-bd02955d419b.png)

#### 하이퍼 파라미터 최적 자동화 

![image](https://user-images.githubusercontent.com/107663853/206637699-9aa22fe8-1100-4ae6-a5ae-175e094040e0.png)

#### 모델 훈련
![image](https://user-images.githubusercontent.com/107663853/206637835-2b54c56f-50e4-4717-b253-1540cfdd8a10.png)

#### 모델 평가 및 예측+ CSV 작성
![image](https://user-images.githubusercontent.com/107663853/206637922-98afd92e-040c-4d9d-ab02-bc953116fe1a.png)

#### 최종 성적
![image](https://user-images.githubusercontent.com/107663853/206638123-0630e108-e701-46e4-a0cf-d119e311ceee.png)
