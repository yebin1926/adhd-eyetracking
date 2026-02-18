# load raw → segment (Win+C3) → PI per segment → save cached .npz/.pt

""" 
Details:
	•	Load raw gaze per student
	•	Sliding window scores (Win) using C3 cost
	•	Peak picking + min distance + min segment length
	•	Segment the series
	•	Compute PI vector (444) per segment
	•	Save cached file per student

"""