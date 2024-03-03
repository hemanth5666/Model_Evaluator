import sys
import pickle
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QMessageBox, \
    QFileDialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


class MLProjectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Machine Learning Project GUI')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        self.data_path_label = QLabel('Data Path:')
        layout.addWidget(self.data_path_label)

        self.data_path_button = QPushButton('Select Data')
        self.data_path_button.clicked.connect(self.open_data_file)
        layout.addWidget(self.data_path_button)

        self.model_path_label = QLabel('Model Path:')
        layout.addWidget(self.model_path_label)

        self.model_path_button = QPushButton('Select Model')
        self.model_path_button.clicked.connect(self.open_model_file)
        layout.addWidget(self.model_path_button)

        self.evaluate_model_button = QPushButton('Evaluate Model')
        self.evaluate_model_button.clicked.connect(self.evaluate_model)
        layout.addWidget(self.evaluate_model_button)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.X_test = None
        self.y_test = None
        self.model = None

    def open_data_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Data File', '', 'Excel Files (*.xlsx)', options=options)
        if file_path:
            try:
                self.data_path_label.setText(f'Data Path: {file_path}')
                self.X_test, self.y_test = self.preprocess_data(file_path)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'An error occurred while loading the data:\n{str(e)}')

    def open_model_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '', 'Model Files (*.pkl)', options=options)
        if file_path:
            self.model_path_label.setText(f'Model Path: {file_path}')
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)

    def preprocess_data(self, file_path):
        try:
            data = pd.read_excel(file_path)

            if 'prognosis' not in data.columns:
                QMessageBox.critical(self, 'Error', "'target_column' not found in the dataset.")
                return None, None

            X = data.drop(columns=['prognosis'])
            y = data['prognosis']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_test, y_test
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while loading the data:\n{str(e)}')
            return None, None

    def evaluate_model(self):
        if self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, 'Warning', 'Please select a data file first.')
            return

        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please select a model file.')
            return

        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            confusion_mat = confusion_matrix(self.y_test, y_pred)

            self.status_label.setText(f'Model Accuracy: {accuracy}')
            self.plot_confusion_matrix(confusion_mat)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred during model evaluation:\n{str(e)}')

    def plot_confusion_matrix(self, confusion_mat):
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()


def main():
    app = QApplication(sys.argv)
    window = MLProjectGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
