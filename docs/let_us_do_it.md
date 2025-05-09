# how to use the cuda kernels
## cudaMemcpy() and cudaMemcpyAsync()
### cudaMemcpy()
* cudaMemcpy()是一个同步函数。这意味着在调用cudaMemcpy()时，CPU会阻塞并等待数据传输完成。只有当数据完全传输完毕后，cudaMemcpy()才会返回，CPU才会继续执行后续的代码。
### cudaMemcpyAsync()
* cudaMemcpyAsync()是一个异步函数。当调用cudaMemcpyAsync()时，CPU不会立即阻塞。函数会迅速返回，CPU可以继续执行后续的代码。数据传输在后台进行，与CPU的执行并行。
* 需要流(streams),由于是异步的，cudaMemcpyAsync()需要与CUDA流一起使用。需要创建创建流，并将cudaMemcpyAsync()操作放入该流中。
* 完成通知，由于cudaMemcpyAsync()是异步的，需要某种方式来确定数据传输何时完成。可以通过以下方式实现：
	* cudaStreamSynchronize(stream):调用此函数会使CPU阻塞，直到指定流中的所有操作(包括cudaMemcpyAsync())完成。
	* 事件(events),可以创建CUDA事件，并在cudaMemcpyAsync()操作完成时记录一个事件。然后，你可以使用cudaEventSynchronize()或cudaEventQuery()来检查事件是否发生，从而确定传输是否完成。

