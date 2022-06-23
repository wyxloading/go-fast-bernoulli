package go_fast_bernoulli

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestFastBernoulli(t *testing.T) {
	var (
		r              = rand.New(rand.NewSource(time.Now().UnixNano()))
		probability    = 0.01
		events         = 10000
		expected       = float64(events) * probability
		errorTolerance = expected * 0.25

		numSampled uint32
		loop       = events
	)
	bernoulli, err := New(probability, r)
	if err != nil {
		t.Fatal(err)
	}
	for {
		if bernoulli.Trial() {
			numSampled++
		}
		loop--
		if loop < 1 {
			break
		}
	}
	var (
		min = uint32(expected - errorTolerance)
		max = uint32(expected + errorTolerance)
	)
	if numSampled < min || numSampled > max {
		t.Fatalf("expected ~%v samples, found %v (acceptable range is %v to %v)", expected, numSampled, min, max)
	}
}

func TestFastBernoulli_Edge(t *testing.T) {
	run := func(fb *FastBernoulli, events int) uint32 {
		var numSampled uint32
		for {
			if fb.Trial() {
				numSampled++
			}
			events--
			if events < 1 {
				break
			}
		}
		return numSampled
	}

	{
		never, err := New(0, rand.New(rand.NewSource(time.Now().UnixNano())))
		if err != nil {
			t.Fatal(err)
		}
		skipCount := never.SkipCount()
		if skipCount != math.MaxUint32 {
			t.Fatalf("never trial instance skipCount should be max uint32, but got %v", skipCount)
		}
		numSampled := run(never, 10000)
		if numSampled != 0 {
			t.Fatalf("expected never sampled, but got %v", numSampled)
		}
	}

	{
		ever, err := New(1, rand.New(rand.NewSource(time.Now().UnixNano())))
		if err != nil {
			t.Fatal(err)
		}
		skipCount := ever.SkipCount()
		if skipCount != 0 {
			t.Fatalf("ever trial instance skipCount should be 0, got %v", skipCount)
		}
		numSampled := run(ever, 10000)
		if numSampled != 10000 {
			t.Fatalf("expected every sampled, but got %v", numSampled)
		}
	}
}
