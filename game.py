import cv2
import mediapipe as mp

mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hand.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7
)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    rgb_image = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hand.HAND_CONNECTIONS)

    cv2.imshow("Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('M'):
        break

camera.release()
cv2.destroyAllWindows()
