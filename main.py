# step1.opencv 라이브러리 불러오기
import cv2

# step2.동영상 파일 열기
cap = cv2.VideoCapture('drive.mp4')

# step3.무한 반복
while True:
    # 동영상 파일 존재 여부(True/False)와 현재 프레임 이미지를 읽음
    retval, frame = cap.read()

    # 만약 동영상 파일이 존재하지 않으면 while 반복문 종료
    if retval == False:
        print("a")
        break

    # 'frame'이란 창 이름으로 현재 프레임 출력
    cv2.imshow('frame', frame)

    # 10초가 지나거나, ESC 키가 입력되면 while 반복문 종료
    if cv2.waitKey(10) == 27:
        break

# step4.동영상 파일 닫고 모든창 종료
cap.release()

cv2.destroyAllWindows()
