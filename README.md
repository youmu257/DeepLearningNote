利用 Keras 建立深度學習模型
===

* 除了基本的 tensorflow 和 keras 外還要裝 h5py  

        pip3 install h5py

* 如果有出現 "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2" 這個警告的話，表示使用基本的 tensorflow 編譯版本，所以沒有支援AVX、AVX2。如果你有GPU可以不用管這個警告，但如果想關閉這個警告的話可以加入以下這行在程式中   

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    但如果你只有CPU的話，你需要重新編譯 tensorflow 去最佳化 CPU版本的支援。[參考](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u?rq=1)

1. 決定輸入的維度，也就是 feature 的數量  

        from keras.layers import Input, Dense
        # 設定輸入為200個 feature
        input = Input(shape=(200,))

2. 決定 hidden layers 層數與 neurons 的數量  
    
        # 兩層 hidden layers 使用 sigmoid activation function
        x = Dense(128, activation='sigmoid')(input)
        x = Dense(256, activation='sigmoid')(x)
        # 最後是輸出層，分類時通常是使用 softmax，讓每個 neurons 總和為1，變成類似機率的值
        output = Dense(5, activation='softmax')(x)

3. 決定模型的 loss function 和 optimizer
    * loss function
        * Regression
            * Mean_squared_error
            * Mean_absolute_error
            * Mean_absolute_percentage_error
            * Mean_squared_logarithmic_error

        * Classification
            * binary_crossentropy
            * categorical_crossentropy

    * optimizer
        * SGD : Stochastic Gradient Descent
        * Adagrad : Adaptive Learning Rate
        * RMSprop : Similar with Adagrad
        * Adam : Similar with RMSprop+Momentum
        * Nadam : Adam+NesterovMomentum

                from keras.models import Model
                # 建立一個 SGD optimizer，設定 learning rate(lr)、momentum、decay、nesterov
                sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
                # 選擇 loss function，輸入剛設定好 optimizer，最後設定 metrics
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


4. 訓練模型

    * trainX     : 訓練資料
    * trainY     : Label(答案)
    * batch_size : 每批資料的大小，訓練完一批權重才會更新
    * verbose    : 是否顯示目前訓練進度，0為不顯示
    * epochs     : 迭代次數，一次迭代表示看過全部 training data
    * shuffle    : 每次迭代後是否打亂(洗牌)
    * validation_split : 0~1,0.1表示一成的測試與九成的訓練資料切割

            history = model.fit( trainX, trainY, batch_size=16, verbose=0, epochs=30, shuffle=True, validation_split=0.1)

5. 結果

        # training part
        loss = history.history.get('loss')
        acc  = history.history.get('acc')
        # validation part
        valLoss = history.history.get('val_loss')
        valAcc = history.history.get('val_acc')

6. 儲存和讀取模型

        # save a model
        model.save('model1.h5')
        # load a model
        # 注意! 要先建立一個一模一樣的框架，load 後還要 compile
        model.load_weights('model1.h5')

7. 預測

        # 預測結果
        pred = model.predict(trainX, verbose=0)


深度學習的流程
===

1. 針對問題選擇合適的 loss function  
    * Regression 常用 mean absolute/squared error  
    * Classification 常用cross-entropy  
2. 設定 learning rate
    * 每次移動的步伐大小，建議設置不同量級去測試，比如0.1，0.01，0.001等
    * 通常不會大於0.1，找出最合適的幸運數字!
3. Activation function

    * Vanishing gradient problem
        * input 被壓縮到一個相對很小的output range
        * Gradient 小，造成無法有效學習
        * Sigmoid, Tanh, Softsign都有這樣的特性
        * 特別不適用於深的深度學習模型
    * Other activation function
        * ReLU, Softplus
        * Hidden layers 通常使用 ReLU
        * Output layer
            * Regression : linear
            * Classification : softmax
4. Optimizer
    * 簡易說明
        * SGD : 隨機梯度下降
        * Adagrad : 每個參數都有不同的learning rate
        * RMSprop : 類似 Adagrad
        * Adam : 類似 RMSprop + Momentum (隨機更新)
        * Nadam : Adam + NesterovMomentum
    * 如何選擇
        * 通常直接選用 Adam
        * Keras 推薦 RNN 使用 RMSProp

下一步
--
* Over-fitting
* 訓練準確率夠高了，但測試時卻慘不忍睹


1. Regularization
    * 限制weights的大小，讓 input 對 output 的影響變小
    * L1 norm 和 L2 norm

            # 在宣告 layer 時加入 W_regularizer
            from keras.regularizers import l1,l2
            input = Input(shape=(200,))
            x = Dense(128, activation='relu', W_regularizer=l2(0.01))(input)
            x = Dense(256, activation='relu', W_regularizer=l2(0.01))(x)
            output = Dense(5, activation='softmax', W_regularizer=l2(kr))(x)

2. Early Stopping
    * monitor: 要監控的performance index
    * patience: 可以容忍連續幾次的不思長進

            from keras.callbacks import EarlyStopping
            earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)
            # 放在 model.fit 的 callbacks 中
            history = model.fit( trainX, trainY, batch_size=16, 
                                 verbose=0, epochs=30, shuffle=True, 
                                 validation_split=0.1, callbacks=[earlyStopping])
3. Dropout
    * 原本為neurons 跟neurons 之間為fully connected
    * 在訓練過程中，隨機拿掉一些連結(weight 設為0)
    * 會造成training performance 變差
        * Error 變大, 每個neuron 修正得越多, 做得越好
    * 不要一開始就加入Dropout!!!!!!
        * Dropout 會讓training performance 變差
        * Dropout 是在避免overfitting，不是萬靈丹
        * 參數少時，regularization

            from keras.layers.core import Dropout
            input = Input(shape=(200,))
            x = Dense(128, activation='relu')(input)
            x = Dropout(.4)(x)

4. Batch normalization
    * 每個input feature 獨立做normalization
    * 利用batch statistics 做normalization而非整份資料
    * 優點
        * 可以解決Gradient vanishing 的問題
        * 可以用比較大的learning rate
        * 加速訓練
        * 取代dropout & regularizes

            from keras.layers import Input, BatchNormalization
            input = Input(shape=(200,))
            x = Dense(128, activation='relu')(input)
            x = Dropout(.4)(x)
            x = BatchNormalization()(x)

Callback function
--
1. lossHistory
    * 我們可以透過繼承 Callback 編寫自己的 callback function，讓他透過類別成員 self.model。

            from keras.callbacks import Callback
            class LossHistory(Callback):
                def on_train_begin(self,logs={}):
                    self.loss=[]
                    self.acc=[]
                    self.val_loss=[]
                    self.val_acc=[]
                def on_batch_end(self,batch,logs={}):
                    self.loss.append(logs.get('loss'))
                    self.acc.append(logs.get('acc'))
                    self.val_loss.append(logs.get('val_loss'))
                    self.val_acc.append(logs.get('val_acc'))

            # 宣告一個自訂的 LossHistory，後續會放入 callbacks 中
            lossHistory = LossHistory()

2. ModelCheckPoint
    * 利用 check point 儲存 model 
    * 第一個參數是儲存模型的名稱
    * monitor : 要監控的類別，{val_acc, val_loss}
    * verbose : 是否顯示目前進度，0為不顯示
    * save_best_only : 設定為 True，只會儲存最佳結果，而不會每次迭代都蓋過去
    * mode : 三種模式 {auto, max, min}，如果監控的類別是 val_acc 那麼就要設定 max，如果是監控是 val_loss 那就要設定 min，auto會自動去偵測要用哪個

            from keras.callbacks import ModelCheckpoint
            checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, 
                                         save_best_only=True, mode='min')
            history = model.fit( trainX, trainY, batch_size=16, 
                                 verbose=0, epochs=30, shuffle=True, 
                                 validation_split=0.1, callbacks=[
                                    earlyStopping, lossHistory, checkpoint])

Reference
==
1. [手把手教你深度學習實務](https://www.slideshare.net/tw_dsconf/ss-83976998)
2. [Keras Document](https://keras.io/callbacks/)