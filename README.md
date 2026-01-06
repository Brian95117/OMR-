# 📄 OMR 自動閱卷系統
Optical Mark Recognition (OMR) Based Automatic Grading System

本專案為一套以 光學劃記辨識（Optical Mark Recognition, OMR） 為基礎之自動閱卷系統，
透過影像處理技術自動辨識答案卡上的填塗選項，並與標準答案比對後完成計分。

本系統不需仰賴昂貴的專用閱卷機設備，僅需一般掃描器即可使用，
適合中小型課程、教師測驗與學術研究用途。

🔍 專案功能特色

✅ 自動產生含 QR Code 與學生姓名 的個人化答案卡

✅ 支援 單選題與多選題

✅ 多選題支援「全對給分」與「部分給分」

✅ 以 Streamlit 建立圖形化操作介面

✅ 支援 PDF 批次作答掃描檔

✅ 不需專用硬體即可進行自動閱卷

🏗 系統架構說明

本專案主要分為兩個子系統：

1️⃣ Answer_sheet_generator（答案卡產生系統）

依學生名單自動產生個人化答案卡

答案卡包含 QR Code 與學生姓名

輸出為可列印之 PDF 檔案

2️⃣ omr-grading-system（OMR 自動閱卷系統）

讀取學生作答掃描檔（PDF）

自動辨識填塗選項

與標準答案比對並計分

輸出學生作答與成績結果

📁 專案資料夾結構
OMR-/
├── Answer_sheet_generator/
│   ├── app.py
│   └── student_list.xlsx
│
├── omr-grading-system/
│   ├── omr_ui.py
│   ├── requirements.txt
│   ├── Scantron.pdf
│   ├── key.xlsx
│   ├── ANS.xlsx
│   └── student_list.xlsx
│
├── LICENSE
└── README.md

🧾 Answer Sheet Generator（答案卡產生）
功能說明

此模組用於將學生名單（Excel）與答案卡模板結合，
自動產生每位學生專屬的答案卡。

答案卡內容包含：

QR Code（內含學生學號）

學生中文姓名（支援中文字型）

使用方式
cd Answer_sheet_generator
streamlit run app.py

🧠 OMR Grading System（自動閱卷系統）
功能說明

此模組為本專案核心，負責：

讀取學生作答掃描 PDF

辨識填塗選項

與標準答案比對

自動計分並輸出結果

使用方式
cd omr-grading-system
pip install -r requirements.txt
streamlit run omr_ui.py

🖼 OMR 影像處理流程

PDF 轉影像（建議 300 DPI）

擷取答案區域（ROI）

建立答案泡泡位置

自動偵測（Hough Circle Transform）

或手動設定泡泡間距

判斷填塗狀態（黑色像素比例）

🧮 計分機制說明
支援題型

單選題（A～E）

多選題（如 AC、BDE）

多選題計分方式

全對給分制：完全正確才給分

部分給分制：

每選對一個選項給 1/k 分

選錯扣 1/k 分

最低分不低於 0 分

🧪 使用建議

建議掃描解析度：300 DPI

建議影像歪斜角度：±2° 以內

若辨識不穩定，請手動調整：

ROI 區域

泡泡間距與大小參數

🚀 未來改進方向

自動歪斜校正（Perspective Transform）

作答塗改與擦除偵測

成績輸出格式客製化

與校務系統或資料庫整合

📝 專案說明

本專案為課程期末專題實作，
旨在結合影像處理與程式設計技術，
實際解決教育場域中自動閱卷的需求問題。
