// [What follows is another outstanding comment from Jim Blandy explaining why
// this technique works.]
//
// This comment should just read, "Generate skip counts with a geometric
// distribution", and leave everyone to go look that up and see why it's the
// right thing to do, if they don't know already.
//
// BUT IF YOU'RE CURIOUS, COMMENTS ARE FREE...
//
// Instead of generating a fresh random number for every trial, we can
// randomly generate a count of how many times we should return false before
// the next time we return true. We call this a "skip count". Once we've
// returned true, we generate a fresh skip count, and begin counting down
// again.
//
// Here's an awesome fact: by exercising a little care in the way we generate
// skip counts, we can produce results indistinguishable from those we would
// get "rolling the dice" afresh for every trial.
//
// In short, skip counts in Bernoulli trials of probability `P` obey a geometric
// distribution. If a random variable `X` is uniformly distributed from
// `[0..1)`, then `floor(log(X) / log(1-P))` has the appropriate geometric
// distribution for the skip counts.
//
// Why that formula?
//
// Suppose we're to return `true` with some probability `P`, say, `0.3`. Spread
// all possible futures along a line segment of length `1`. In portion `P` of
// those cases, we'll return true on the next call to `trial`; the skip count is
// 0. For the remaining portion `1-P` of cases, the skip count is `1` or more.
//
// ```
//    skip:             0                         1 or more
//             |------------------^-----------------------------------------|
// portion:            0.3                            0.7
//                      P                             1-P
// ```
//
// But the "1 or more" section of the line is subdivided the same way: *within
// that section*, in portion `P` the second call to `trial()` returns `true`, and
// in portion `1-P` it returns `false` a second time; the skip count is two or
// more. So we return `true` on the second call in proportion `0.7 * 0.3`, and
// skip at least the first two in proportion `0.7 * 0.7`.
//
// ```
//    skip:             0                1              2 or more
//             |------------------^------------^----------------------------|
// portion:            0.3           0.7 * 0.3          0.7 * 0.7
//                      P             (1-P)*P            (1-P)^2
// ```
//
// We can continue to subdivide:
//
// ```
// skip >= 0:  |------------------------------------------------- (1-P)^0 --|
// skip >= 1:  |                  ------------------------------- (1-P)^1 --|
// skip >= 2:  |                               ------------------ (1-P)^2 --|
// skip >= 3:  |                                 ^     ---------- (1-P)^3 --|
// skip >= 4:  |                                 .            --- (1-P)^4 --|
//                                               .
//                                               ^X, see below
// ```
//
// In other words, the likelihood of the next `n` calls to `trial` returning
// `false` is `(1-P)^n`. The longer a run we require, the more the likelihood
// drops. Further calls may return `false` too, but this is the probability
// we'll skip at least `n`.
//
// This is interesting, because we can pick a point along this line segment and
// see which skip count's range it falls within; the point `X` above, for
// example, is within the ">= 2" range, but not within the ">= 3" range, so it
// designates a skip count of `2`. So if we pick points on the line at random
// and use the skip counts they fall under, that will be indistinguishable from
// generating a fresh random number between `0` and `1` for each trial and
// comparing it to `P`.
//
// So to find the skip count for a point `X`, we must ask: To what whole power
// must we raise `1-P` such that we include `X`, but the next power would
// exclude it? This is exactly `floor(log(X) / log(1-P))`.
//
// Our algorithm is then, simply: When constructed, compute an initial skip
// count. Return `false` from `trial` that many times, and then compute a new
// skip count.
//
// For a call to `MultiTrial(n)`, if the skip count is greater than `n`, return
// `false` and subtract `n` from the skip count. If the skip count is less than
// `n`, return true and compute a new skip count. Since each trial is
// independent, it doesn't matter by how much `n` overshoots the skip count; we
// can actually compute a new skip count at *any* time without affecting the
// distribution. This is really beautiful.

package go_fast_bernoulli

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// New construct a new `FastBernoulli` instance that samples events with the
// given probability.
func New(probability float64, r *rand.Rand) (*FastBernoulli, error) {
	if probability < 0 || probability > 1 {
		return nil, fmt.Errorf("invalid probability")
	}
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	f := &FastBernoulli{
		probability:          probability,
		invLogNotProbability: 1 / math.Log(1.0-probability),
		r:                    r,
	}
	f.resetSkipCount()
	return f, nil
}

// FastBernoulli sampling: each event has equal probability of being sampled.
type FastBernoulli struct {
	// The likelihood that any given call to `Trial` should return true.
	probability float64
	// The value of 1 / math.Log(1 - probability), cached for repeated use.
	//
	// If probability is exactly 0 or exactly 1, we don't use this value.
	// Otherwise, we guarantee this value is in the range [-2**53, -1/37), i.e.
	// definitely negative, as required by chooseSkipCount. See setProbability for
	// the details.
	invLogNotProbability float64
	r                    *rand.Rand

	skipCount uint32
}

// Trial perform a Bernoulli trial: Return `true` with the configured probability
//
// Call this each time an event occurs to determine whether to sample the event.
//
// The lower the configured probability, the less overhead calling this function has.
func (f *FastBernoulli) Trial() bool {
	if f.skipCount > 0 {
		f.skipCount--
		return false
	}
	f.resetSkipCount()
	return f.probability != 0
}

// MultiTrial Perform `n` Bernoulli trials at once.
//
// This is semantically equivalent to calling the `Trial()` method `n`
// times and returning `true` if any of those calls returned `true`, but
// runs in `O(1)` time instead of `O(n)` time.
//
// What is this good for? In some applications, some events are "bigger"
// than others. For example, large memory allocations are more significant
// than small memory allocations. Perhaps we'd like to imagine that we're
// drawing allocations from a stream of bytes, and performing a separate
// Bernoulli trial on every byte from the stream. We can accomplish this by
// calling `MultiTrial(s)` for the number of bytes `s`, and sampling the
// event if that call returns true.
//
// Of course, this style of sampling needs to be paired with analysis and
// presentation that makes the "size" of the event apparent, lest trials
// with large values for `n` appear to be indistinguishable from those with
// small values for `n`, despite being potentially much more likely to be
// sampled.
func (f *FastBernoulli) MultiTrial(n uint32) bool {
	if n < f.skipCount {
		f.skipCount -= n
		return false
	}
	f.resetSkipCount()
	return f.probability != 0
}

// Probability get the probability with which events are sampled.
//
// This is a number between `0.0` and `1.0`.
//
// This is the same value that was passed to `New` when constructing this
// instance.
func (f *FastBernoulli) Probability() float64 {
	return f.probability
}

// SkipCount return how many events will be skipped until the next event is sampled
//
// When `Probability() == 0` this method's return value is
// inaccurate, and logically should be infinity.
func (f *FastBernoulli) SkipCount() uint32 {
	return f.skipCount
}

func (f *FastBernoulli) resetSkipCount() {
	switch f.probability {
	case 0: // Edge case: we will never sample any event.
		f.skipCount = math.MaxUint32
	case 1: // Edge case: we will sample every event.
		f.skipCount = 0
	default:
		// Common case: we need to choose a new skip count using the
		// formula `floor(log(x) / log(1 - P))`, as explained in the
		// comment at the top of this file.
		x := f.r.Float64()
		skipCount := math.Floor(math.Log(x) * f.invLogNotProbability)
		if skipCount <= math.MaxUint32 {
			f.skipCount = uint32(skipCount)
		} else {
			f.skipCount = math.MaxUint32
		}
	}
}
