import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup data directories
train_dir = 'dataset/train'

# Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32,
                                    class_mode='categorical', subset='training')
val = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32,
                                  class_mode='categorical', subset='validation')

# Load MobileNetV2
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train.num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train for 1 epoch just to generate the model
model.fit(train, validation_data=val, epochs=1)

# ✅ Save it
model.save("smart_sorting_model.keras")

print("🎉 Model saved successfully as smart_sorting_model.keras")
